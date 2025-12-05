import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from plotter import *


class Expando:
    """A class that allows dynamic attribute assignment."""
    pass

class InputData:

    def __init__(self,
                 variables: list,
                 gas_prices: list[float],
                 coal_prices: list[float],
                 eua_prices: list[float],
                 rhs_demand: float,
                 rhs_storage: dict[float],
                 rhs_prod: dict[float],
                 rhs_prod_wind: list[float],
                 rhs_prod_pv: list[float],
                 efficiencies: dict[float],
                 co2_per_kWh: dict[float],
                 min_prod_ratio: dict[float],
                 starting_storage_levels: dict[float],
                 starting_eua_balance: float):
        
        self.variables = variables
        self.gas_prices = gas_prices
        self.coal_prices = coal_prices
        self.eua_prices = eua_prices
        self.rhs_demand = rhs_demand
        self.rhs_storage = rhs_storage
        self.rhs_prod = rhs_prod
        self.rhs_wind_prod = rhs_prod_wind
        self.rhs_pv_prod = rhs_prod_pv
        self.efficiencies = efficiencies
        self.co2_per_kWh = co2_per_kWh
        self.min_prod_ratio = min_prod_ratio
        self.starting_storage_levels = starting_storage_levels
        self.starting_eua_balance = starting_eua_balance
        

class DeterministicModel():

    def __init__(self, input_data: InputData, name: str = "Deterministic Optimization Model", days: int = 366):
        self.data = input_data
        # keep numeric day count and an iterable range for loops
        self.n_days = int(days)
        self.days = range(self.n_days)
        self.name = name
        self.results = Expando()
        self._build_model()

    def _build_variables(self):
        # build a dict mapping variable name -> list(vars over time)
        self.variables = {
            v: [self.model.addVar(name=f"{v}_{t}") for t in self.days]
            for v in self.data.variables
        }
    
    def _build_constraints(self):

        # Annual demand constraint
        self.demand = self.model.addLConstr(
            gp.quicksum(self.variables['P_COAL'][t] + self.variables['P_GAS'][t] + self.variables['P_WIND'][t] + self.variables['P_PV'][t] for t in self.days),
                         GRB.GREATER_EQUAL, self.data.rhs_demand)
    
        # Storage maximum capacity constraints
        self.storage_gas__max = [self.model.addLConstr(
            self.variables['Q_GAS_STORAGE'][t], GRB.LESS_EQUAL, self.data.rhs_storage['Q_GAS_STORAGE']
            )   
            for t in self.days
        ]
        
        self.storage_coal__max = [self.model.addLConstr(
            self.variables['Q_COAL_STORAGE'][t], GRB.LESS_EQUAL, self.data.rhs_storage['Q_COAL_STORAGE']
            )
            for t in self.days
        ]
        
        # Storage balance constraints
        self.storage_gas = [self.model.addLConstr(
             self.variables['Q_GAS_BUY'][t] + self.variables['Q_GAS_STORAGE'][t-1] - (self.variables['P_GAS'][t] / self.data.efficiencies['eta_GAS']), GRB.EQUAL, self.variables['Q_GAS_STORAGE'][t]
            )
            for t in range(1, self.n_days)
        ]
                        
        
        self.storage_coal = [self.model.addLConstr(
             self.variables['Q_COAL_BUY'][t] + self.variables['Q_COAL_STORAGE'][t-1] - (self.variables['P_COAL'][t] / self.data.efficiencies['eta_COAL']), GRB.EQUAL, self.variables['Q_COAL_STORAGE'][t] 
            )
            for t in range(1, self.n_days)
        ]
        
        # Initial storage level constraints
        self.init_storage_gas = self.model.addLConstr(
            self.variables['Q_GAS_BUY'][0] + self.data.starting_storage_levels['Q_GAS_STORAGE'] - (self.variables['P_GAS'][0] / self.data.efficiencies['eta_GAS']), GRB.EQUAL, self.variables['Q_GAS_STORAGE'][0]
        )
        
        self.init_storage_coal = self.model.addLConstr(
            self.variables['Q_COAL_BUY'][0] + self.data.starting_storage_levels['Q_COAL_STORAGE'] - (self.variables['P_COAL'][0] / self.data.efficiencies['eta_COAL']), GRB.EQUAL, self.variables['Q_COAL_STORAGE'][0]
        )
        
        # Final storage level constraints
        self.final_storage_gas = self.model.addLConstr(
            self.variables['Q_GAS_STORAGE'][self.n_days - 1], GRB.EQUAL, self.data.starting_storage_levels['Q_GAS_STORAGE']
        )
        
        self.final_storage_coal = self.model.addLConstr(
            self.variables['Q_COAL_STORAGE'][self.n_days - 1], GRB.EQUAL, self.data.starting_storage_levels['Q_COAL_STORAGE']
        )
        
       # CO2 Emission per scenario
        self.CO2_emission = self.model.addLConstr(
            gp.quicksum(
                self.variables['P_GAS'][t] * self.data.co2_per_kWh['CO2_per_kWh_GAS']
                + self.variables['P_COAL'][t] * self.data.co2_per_kWh['CO2_per_kWh_COAL']
                for t in self.days
            ),
            GRB.EQUAL,
            self.variables['Q_EUA_BALANCE'][self.n_days - 1]
        )
        
        self.EUA_balance = [self.model.addLConstr(
            self.variables['Q_EUA_BALANCE'][t-1] + self.variables['Q_EUA_BUY'][t] - self.variables['Q_EUA_SELL'][t],
            GRB.EQUAL,
            self.variables['Q_EUA_BALANCE'][t]
        ) for t in range(1, self.n_days)]
        
        self.EUA_balance_init = self.model.addLConstr(
            self.variables['Q_EUA_BALANCE'][0],
            GRB.EQUAL,
            self.data.starting_eua_balance
        )
        
        self.EUA_max_sell = [self.model.addLConstr(
            self.variables['Q_EUA_SELL'][t], GRB.LESS_EQUAL, 1_000_000
        ) for t in self.days]
        
        self.EUA_max_buy = [self.model.addLConstr(
            self.variables['Q_EUA_BUY'][t], GRB.LESS_EQUAL, 1_000_000
        ) for t in self.days]
        
        
        # Maximum production constraints
        self.max_prod_COAL_cap = [ self.model.addLConstr(
            self.variables['P_COAL'][t], GRB.LESS_EQUAL, self.data.rhs_prod['P_COAL']
        )
        for t in self.days
        ]
        
        self.max_prod_GAS_cap = [ self.model.addLConstr(
            self.variables['P_GAS'][t], GRB.LESS_EQUAL, self.data.rhs_prod['P_GAS']
        )
        for t in self.days
        ]
        
        # Maximum production constrainted by resource availability
        self.max_prod_GAS = [ self.model.addLConstr(
            self.variables['P_GAS'][t], GRB.LESS_EQUAL, self.data.efficiencies['eta_GAS'] * (self.variables['Q_GAS_STORAGE'][t-1] + self.variables['Q_GAS_BUY'][t])
        ) 
            for t in range(1, self.n_days)
        ]
        
        self.max_prod_COAL = [ self.model.addLConstr(
            self.variables['P_COAL'][t], GRB.LESS_EQUAL, self.data.efficiencies['eta_COAL'] * (self.variables['Q_COAL_STORAGE'][t-1] + self.variables['Q_COAL_BUY'][t])
        ) 
            for t in range(1, self.n_days)
        ]
        
        # Initial maximum production constrainted by resource availability
        self.max_prod_GAS_init = self.model.addLConstr(
            self.variables['P_GAS'][0], GRB.LESS_EQUAL, self.data.efficiencies['eta_GAS'] * (self.data.starting_storage_levels['Q_GAS_STORAGE'] + self.variables['Q_GAS_BUY'][0])
        )
        
        self.max_prod_COAL_init = self.model.addLConstr(
            self.variables['P_COAL'][0], GRB.LESS_EQUAL, self.data.efficiencies['eta_COAL'] * (self.data.starting_storage_levels['Q_COAL_STORAGE'] + self.variables['Q_COAL_BUY'][0])
        )
        
        self.max_prod_wind = [ self.model.addLConstr(
            self.variables['P_WIND'][t], GRB.LESS_EQUAL, self.data.rhs_wind_prod[t]
        )
        for t in self.days
        ]  
        
        self.max_prod_pv = [ self.model.addLConstr(
            self.variables['P_PV'][t], GRB.LESS_EQUAL, self.data.rhs_pv_prod[t]
        )
        for t in self.days
        ]
        
        
        # Minimum production for coal and gas plants
        self.min_prod_COAL = [ self.model.addLConstr(
            self.variables['P_COAL'][t], GRB.GREATER_EQUAL, self.data.min_prod_ratio['min_prod_ratio_COAL'] * self.data.rhs_prod['P_COAL']
        )
        for t in self.days
        ]
        
        self.min_prod_GAS = [ self.model.addLConstr(
            self.variables['P_GAS'][t], GRB.GREATER_EQUAL, self.data.min_prod_ratio['min_prod_ratio_GAS'] * self.data.rhs_prod['P_GAS']
        )
        for t in self.days
        ]
        

        ### All variables are automtically set to be greater than or equal to zero



    def _build_objective(self):
        # objective: minimise fuel & EUA purchase costs over time
        self.model.setObjective(
            gp.quicksum(
                self.variables['Q_GAS_BUY'][t] * self.data.gas_prices[t]
                + self.variables['Q_COAL_BUY'][t] * self.data.coal_prices[t]
                + (self.variables['Q_EUA_BUY'][t]- self.variables['Q_EUA_SELL'][t]) * self.data.eua_prices[t]
                for t in self.days
            ),
            GRB.MINIMIZE
        )


    def _build_model(self):
        self.model = gp.Model(self.name)
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()
    
    
    def _save_results(self):
        self.results.obj_val = self.model.ObjVal
        # save variable values as dict keyed by (var_name, t)
        self.results.var_vals = {(v, t): self.variables[v][t].x
                                 for v in self.data.variables for t in self.days}
        # please return the dual values for all constraints
        # self.results.dual_vals = {
        #     'upper_power': [self.upper_power[i].Pi for i in range(len(self.upper_power))],
        #     'hourly_balance': [self.hourly_balance[i].Pi for i in range(len(self.hourly_balance))],
        #     'daily_balance': self.daily_balance.Pi
        # }

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise ValueError("No optimal solution found for {self.model.name}")
        
    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Optimal objective value:")
        print(self.results.obj_val)
        print("Optimal variable values:")
        #print(self.results.var_vals)
        #print("Optimal dual values:")
        #print(self.results.dual_vals)


    def plot_results(self):
        
        plotting_days = self.days 
        
        # color palette
        colors, background_color = color_palette() 
    
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.step(
            plotting_days,
            [self.results.var_vals[('P_GAS', t)] for t in plotting_days],
            label='P_GAS',
            where='mid', color=colors[0]
        )

        ax.step(
            plotting_days,
            [self.results.var_vals[('P_COAL', t)] for t in plotting_days],
            label='P_COAL',
            where='mid', color=colors[2]
        )
        
        ax.step(
            plotting_days,
            [self.results.var_vals[('P_WIND', t)] for t in plotting_days],
            label='P_WIND',
            where='mid', color=colors[4]
        )
        
        ax.step(
            plotting_days,
            [self.results.var_vals[('P_PV', t)] for t in plotting_days],
            label='P_PV',
            where='mid', color = colors[6]
        )
        
        ax.set_xlabel('Day')
        ax.text(0.0, 1.07, 'Optimal Power Production Over Time', transform=ax.transAxes, fontsize=14, color='black', ha ='left', fontweight='bold')
        ax.text(0.0, 1.03, 'kWh production for each power generating unit for days 225 to 274', transform=ax.transAxes, fontsize=10, color='black', ha ='left')
        ax.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        ax.legend()
        ax.grid()
        plt.show()
        
        # Plots of storage levels
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.step(plotting_days,[self.results.var_vals[('Q_GAS_STORAGE', t)] for t in plotting_days], label='Gas Storage Level', where='mid', color=colors[1])
        ax.step(plotting_days,[self.results.var_vals[('Q_COAL_STORAGE', t)] for t in plotting_days], label='Coal Storage Level', where='mid', color=colors[3])
        ax.set_xlabel('Day')
        ax.text(0.0, 1.07, 'Optimal Fuel Storage Levels Over Time', transform=ax.transAxes, fontsize=14, color='black', ha ='left', fontweight='bold')
        ax.text(0.0, 1.03, 'kWh fuel storage levels for gas and coal storage for days 225 to 274', transform=ax.transAxes, fontsize=10, color='black', ha ='left')
        ax.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        ax.legend()
        ax.grid()
        plt.show()
        
        # Plot of purchase quantities
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.step(plotting_days,[self.results.var_vals[('Q_GAS_BUY', t)] for t in plotting_days], label='Gas Purchase Quantity', where='mid', color=colors[2])
        ax.step(plotting_days,[self.results.var_vals[('Q_COAL_BUY', t)] for t in plotting_days], label='Coal Purchase Quantity', where='mid', color=colors[4])
        ax.set_xlabel('Day')
        ax.text(0.0, 1.07, 'Optimal Fuel Purchase Quantities Over Time', transform=ax.transAxes, fontsize=14, color='black', ha ='left', fontweight='bold')
        ax.text(0.0, 1.03, 'kWh fuel purchase quantities for gas and coal for days 225 to 274', transform=ax.transAxes, fontsize=10, color='black', ha ='left')
        ax.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        ax.legend()
        ax.grid()
        plt.show() 
        
        # Plot buy and sell of EUAs for selected days
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.step(plotting_days,[self.results.var_vals[('Q_EUA_BUY', t)] for t in plotting_days], label='EUA Purchase Quantity', where='mid', color=colors[0])
        ax.step(plotting_days,[self.results.var_vals[('Q_EUA_SELL', t)] for t in plotting_days], label='EUA Sell Quantity', where='mid', color=colors[3])
        ax.set_xlabel('Day')
        ax.text(0.0, 1.07, 'Optimal EUA Purchase and Sell Quantities Over Time', transform=ax.transAxes, fontsize=14, color='black', ha ='left', fontweight='bold')
        ax.text(0.0, 1.03, 'kWh EUA purchase and sell quantities for days 225 to 274', transform=ax.transAxes, fontsize=10, color='black', ha ='left')
        ax.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        ax.legend()
        ax.grid()
        plt.show()