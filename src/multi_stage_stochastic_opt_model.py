import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import random
import seaborn as sns
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

 
 
class StochasticModel_Second_Stage():
    def __init__(self, input_data: InputData, name: str = "Stochastic Optimization Model", days: int = 366, n_scenario: int = 10, risk_averse: bool = False, alpha: float = 0.95, beta: float = 0.5, stage: int = 2, no_stages: int = 4, first_stage_results: dict = None):
        self.data = input_data
        self.first_stage = first_stage_results
        # keep numeric day count and an iterable range for loops
        self.n_days = int(days)
        self.stage = stage
        self.no_stages = no_stages
        self.split = int(days*((stage-1)/no_stages))
        self.days = range(self.n_days)
        self.n_scenario = range(n_scenario)
        self.risk_averse = risk_averse
        self.alpha = alpha
        self.beta = beta
        self.name = name
        self.results = Expando()
        self.scenarios = self.generate_scenarios(n_scenario, sampling_method='normal')
        self._build_model()
    
    def generate_scenarios(self, n_scenarios: int, sampling_method: str = 'normal'):
        scenarios = []
        
        def uniform_sample(lower=0.8, upper=1.2):
            return random.uniform(lower, upper)

        def normal_sample(mu=0.0, sigma=0.1, lower=0.8, upper=1.2):

            return max(lower, min(upper, 1 + random.gauss(mu, sigma)))

        for s in range(n_scenarios):
            # Use one shared daily shock to keep prices/renewables correlated across drivers
            if sampling_method == 'normal':
                daily_shocks = [normal_sample() for _ in self.days]
            elif sampling_method == 'uniform':
                daily_shocks = [uniform_sample() for _ in self.days]
            else:
                raise ValueError(f"Unsupported sampling method: {sampling_method}")

            # Cap renewables at daily capacity, e.g., 100,000 kW
            capped_wind = [min(prod * daily_shocks[t], 24 * 150_000) for t, prod in enumerate(self.data.rhs_wind_prod)]
            capped_pv = [min(prod * daily_shocks[t], 24 * 100_000) for t, prod in enumerate(self.data.rhs_pv_prod)]
            
            
            
            gas_prices = self.data.gas_prices[:self.split]
                
            for t, price in enumerate(self.data.gas_prices):
                if t >= self.split:
                    gas_prices.append(price * daily_shocks[t])
            
            coal_prices = self.data.coal_prices[:self.split]
            for t, price in enumerate(self.data.coal_prices):
                if t >= self.split:
                    coal_prices.append(price * daily_shocks[t])
            
            eua_prices = self.data.eua_prices[:self.split]
            for t, price in enumerate(self.data.eua_prices):
                if t >= self.split:
                    eua_prices.append(price * daily_shocks[t])
            
            rhs_prod_wind = self.data.rhs_wind_prod[:self.split]
            for t, prod in enumerate(self.data.rhs_wind_prod):
                if t >= self.split:
                    rhs_prod_wind.append(min(prod * daily_shocks[t], 24 * 150_000))
            
            rhs_prod_pv = self.data.rhs_pv_prod[:self.split]
            for t, prod in enumerate(self.data.rhs_pv_prod):
                if t >= self.split:
                    rhs_prod_pv.append(min(prod * daily_shocks[t], 24 * 100_000))
                    
            
            scenario_data = InputData(
                variables=self.data.variables,
                gas_prices=gas_prices,
                coal_prices=coal_prices,
                eua_prices=eua_prices,
                rhs_demand=self.data.rhs_demand,
                rhs_storage=self.data.rhs_storage,
                rhs_prod=self.data.rhs_prod,
                rhs_prod_wind=rhs_prod_wind,
                rhs_prod_pv=rhs_prod_pv,
                efficiencies=self.data.efficiencies,
                co2_per_kWh=self.data.co2_per_kWh,
                min_prod_ratio=self.data.min_prod_ratio,
                starting_storage_levels=self.data.starting_storage_levels,
                starting_eua_balance=self.data.starting_eua_balance
            )
            scenarios.append(scenario_data)
        # Split into in sample and out of sample scenarios
        split_index = int(0.8 * n_scenarios)
        self.out_of_sample_scenarios = scenarios[:split_index]
        self.in_sample_scenarios = scenarios[split_index:]
        
        return self.in_sample_scenarios, self.out_of_sample_scenarios
   
    def _build_variables(self):
        
        # Decsion variables
        self.variables = {
            v: [self.model.addVar(name=f"{v}_{t}") for t in self.days]
            for v in self.data.variables
        }
        # If risk averse is true add CVaR auxillary variables
        if self.risk_averse == True:
            self.zeta = self.model.addVar(name="zeta")
            # One eta per scenario captures excess loss over zeta
            self.eta = {i: self.model.addVar(name=f"eta_scen_{i}") for i in range(len(self.in_sample_scenarios))}
    
    
    def _build_constraints(self):

        # Annual demand must hold for each scenario (robust feasibility)
        self.demand = [self.model.addLConstr(
            gp.quicksum(
                self.variables['P_COAL'][t] + self.variables['P_GAS'][t]
                + self.variables['P_WIND'][t] + self.variables['P_PV'][t]
                for t in self.days
            ),
            GRB.GREATER_EQUAL,
            scen.rhs_demand
        ) for scen in self.in_sample_scenarios]

        # Storage maximum capacity per scenario
        self.storage_gas__max = [self.model.addLConstr(
            self.variables['Q_GAS_STORAGE'][t], GRB.LESS_EQUAL, scen.rhs_storage['Q_GAS_STORAGE']
        ) for t in self.days for scen in self.in_sample_scenarios]

        self.storage_coal__max = [self.model.addLConstr(
            self.variables['Q_COAL_STORAGE'][t], GRB.LESS_EQUAL, scen.rhs_storage['Q_COAL_STORAGE']
        ) for t in self.days for scen in self.in_sample_scenarios]  
        
        # Storage balance per scenario
        self.storage_gas = [self.model.addLConstr(
            self.variables['Q_GAS_BUY'][t] + self.variables['Q_GAS_STORAGE'][t-1]
            - (self.variables['P_GAS'][t] / scen.efficiencies['eta_GAS']),
            GRB.EQUAL,
            self.variables['Q_GAS_STORAGE'][t]
        ) for t in range(1, self.n_days) for scen in self.in_sample_scenarios]

        self.storage_coal = [self.model.addLConstr(
            self.variables['Q_COAL_BUY'][t] + self.variables['Q_COAL_STORAGE'][t-1]
            - (self.variables['P_COAL'][t] / scen.efficiencies['eta_COAL']),
            GRB.EQUAL,
            self.variables['Q_COAL_STORAGE'][t]
        ) for t in range(1, self.n_days) for scen in self.in_sample_scenarios]

        # Initial storage level per scenario
        self.init_storage_gas = [self.model.addLConstr(
            self.variables['Q_GAS_BUY'][0] + scen.starting_storage_levels['Q_GAS_STORAGE']
            - (self.variables['P_GAS'][0] / scen.efficiencies['eta_GAS']),
            GRB.EQUAL,
            self.variables['Q_GAS_STORAGE'][0]
        ) for scen in self.in_sample_scenarios]

        self.init_storage_coal = [self.model.addLConstr(
            self.variables['Q_COAL_BUY'][0] + scen.starting_storage_levels['Q_COAL_STORAGE']
            - (self.variables['P_COAL'][0] / scen.efficiencies['eta_COAL']),
            GRB.EQUAL,
            self.variables['Q_COAL_STORAGE'][0]
        ) for scen in self.in_sample_scenarios]

        # Final storage level per scenario (return to start)
        self.final_storage_gas = [self.model.addLConstr(
            self.variables['Q_GAS_STORAGE'][self.n_days - 1], GRB.EQUAL, scen.starting_storage_levels['Q_GAS_STORAGE']
        ) for scen in self.in_sample_scenarios]

        self.final_storage_coal = [self.model.addLConstr(
            self.variables['Q_COAL_STORAGE'][self.n_days - 1], GRB.EQUAL, scen.starting_storage_levels['Q_COAL_STORAGE']
        ) for scen in self.in_sample_scenarios]
        
        # CO2 Emission per scenario
        self.CO2_emission = [self.model.addLConstr(
            gp.quicksum(
                self.variables['P_GAS'][t] * scen.co2_per_kWh['CO2_per_kWh_GAS']
                + self.variables['P_COAL'][t] * scen.co2_per_kWh['CO2_per_kWh_COAL']
                for t in self.days
            ),
            GRB.EQUAL,
            self.variables['Q_EUA_BALANCE'][self.n_days - 1]
        ) for scen in self.in_sample_scenarios]
        
        self.EUA_balance = [self.model.addLConstr(
            self.variables['Q_EUA_BALANCE'][t-1] + self.variables['Q_EUA_BUY'][t] - self.variables['Q_EUA_SELL'][t],
            GRB.EQUAL,
            self.variables['Q_EUA_BALANCE'][t]
        ) for t in range(1, self.n_days) for scen in self.in_sample_scenarios]
        
        self.EUA_balance_init = [self.model.addLConstr(
            self.variables['Q_EUA_BALANCE'][0],
            GRB.EQUAL,
            self.data.starting_eua_balance
        ) for scen in self.in_sample_scenarios]
        
        self.EUA_max_sell = [self.model.addLConstr(
            self.variables['Q_EUA_SELL'][t], GRB.LESS_EQUAL, 1_000_000
        ) for t in self.days for scen in self.in_sample_scenarios]
        
        self.EUA_max_buy = [self.model.addLConstr(
            self.variables['Q_EUA_BUY'][t], GRB.LESS_EQUAL, 1_000_000
        ) for t in self.days for scen in self.in_sample_scenarios]
        
        # Maximum production capacity per scenario
        self.max_prod_COAL_cap = [self.model.addLConstr(
            self.variables['P_COAL'][t], GRB.LESS_EQUAL, scen.rhs_prod['P_COAL']
        ) for t in self.days for scen in self.in_sample_scenarios]

        self.max_prod_GAS_cap = [self.model.addLConstr(
            self.variables['P_GAS'][t], GRB.LESS_EQUAL, scen.rhs_prod['P_GAS']
        ) for t in self.days for scen in self.in_sample_scenarios]
        
        # Production limited by available fuel per scenario
        self.max_prod_GAS = [self.model.addLConstr(
            self.variables['P_GAS'][t], GRB.LESS_EQUAL,
            scen.efficiencies['eta_GAS'] * (self.variables['Q_GAS_STORAGE'][t-1] + self.variables['Q_GAS_BUY'][t])
        ) for t in range(1, self.n_days) for scen in self.in_sample_scenarios]

        self.max_prod_COAL = [self.model.addLConstr(
            self.variables['P_COAL'][t], GRB.LESS_EQUAL,
            scen.efficiencies['eta_COAL'] * (self.variables['Q_COAL_STORAGE'][t-1] + self.variables['Q_COAL_BUY'][t])
        ) for t in range(1, self.n_days) for scen in self.in_sample_scenarios]

        # Initial production limited by initial storage and buy
        self.max_prod_GAS_init = [self.model.addLConstr(
            self.variables['P_GAS'][0], GRB.LESS_EQUAL,
            scen.efficiencies['eta_GAS'] * (scen.starting_storage_levels['Q_GAS_STORAGE'] + self.variables['Q_GAS_BUY'][0])
        ) for scen in self.in_sample_scenarios]

        self.max_prod_COAL_init = [self.model.addLConstr(
            self.variables['P_COAL'][0], GRB.LESS_EQUAL,
            scen.efficiencies['eta_COAL'] * (scen.starting_storage_levels['Q_COAL_STORAGE'] + self.variables['Q_COAL_BUY'][0])
        ) for scen in self.in_sample_scenarios]

        self.max_prod_wind = [self.model.addLConstr(
            self.variables['P_WIND'][t], GRB.LESS_EQUAL, scen.rhs_wind_prod[t]
        ) for t in self.days for scen in self.in_sample_scenarios]
        
        self.max_prod_pv = [self.model.addLConstr(
            self.variables['P_PV'][t], GRB.LESS_EQUAL, scen.rhs_pv_prod[t]
        ) for t in self.days for scen in self.in_sample_scenarios]

        # Minimum production for coal and gas plants per scenario
        self.min_prod_COAL = [self.model.addLConstr(
            self.variables['P_COAL'][t], GRB.GREATER_EQUAL,
            scen.min_prod_ratio['min_prod_ratio_COAL'] * scen.rhs_prod['P_COAL']
        ) for t in self.days for scen in self.in_sample_scenarios]
    
        self.min_prod_GAS = [self.model.addLConstr(
            self.variables['P_GAS'][t], GRB.GREATER_EQUAL,
            scen.min_prod_ratio['min_prod_ratio_GAS'] * scen.rhs_prod['P_GAS']
        ) for t in self.days for scen in self.in_sample_scenarios]
        
        # Non-Anticipativity constraints
        self.non_anti_P_COAL = [self.model.addLConstr(
            self.variables['P_COAL'][t], GRB.EQUAL,
            self.first_stage[('P_COAL', t)]
        ) for t in range(0,self.split)]

        self.non_anti_P_GAS = [self.model.addLConstr(
            self.variables['P_GAS'][t], GRB.EQUAL,
            self.first_stage[('P_GAS', t)]
        ) for t in range(0,self.split)]  

        self.non_anti_P_COAL = [self.model.addLConstr(
            self.variables['P_COAL'][t], GRB.EQUAL,
            self.first_stage[('P_COAL', t)]
        ) for t in range(0,self.split)]   
            
        self.non_anti_Q_COAL_STORAGE = [self.model.addLConstr(
            self.variables['Q_COAL_STORAGE'][t], GRB.EQUAL,
            self.first_stage[('Q_COAL_STORAGE', t)]
        ) for t in range(0,self.split)]  
        
        self.non_anti_Q_GAS_STORAGE = [self.model.addLConstr(
            self.variables['Q_GAS_STORAGE'][t], GRB.EQUAL,
            self.first_stage[('Q_GAS_STORAGE', t)]
        ) for t in range(0,self.split)]   
        
        self.non_anti_Q_COAL_BUY = [self.model.addLConstr(
            self.variables['Q_COAL_BUY'][t], GRB.EQUAL,
            self.first_stage[('Q_COAL_BUY', t)]
        ) for t in range(0,self.split)]   
        
        self.non_anti_Q_GAS_BUY = [self.model.addLConstr(
            self.variables['Q_GAS_BUY'][t], GRB.EQUAL,
            self.first_stage[('Q_GAS_BUY', t)]
        ) for t in range(0,self.split)]
        
        self.non_anti_Q_EUA_BUY = [self.model.addLConstr(
            self.variables['Q_EUA_BUY'][t], GRB.EQUAL,
            self.first_stage[('Q_EUA_BUY', t)]
        ) for t in range(0,self.split)]
        
        self.non_anti_Q_EUA_SELL = [self.model.addLConstr(
            self.variables['Q_EUA_SELL'][t], GRB.EQUAL,
            self.first_stage[('Q_EUA_SELL', t)]
        ) for t in range(0,self.split)]
        
        self.non_anti_EUA_BALANCE = [self.model.addLConstr(
            self.variables['Q_EUA_BALANCE'][t], GRB.EQUAL,
            self.first_stage[('Q_EUA_BALANCE', t)]
        ) for t in range(0,self.split)]
        
        self.non_anti_P_WIND = [self.model.addLConstr(
            self.variables['P_WIND'][t], GRB.EQUAL,
            self.first_stage[('P_WIND', t)]
        ) for t in range(0,self.split)]
        
        self.non_anti_P_PV = [self.model.addLConstr(
            self.variables['P_PV'][t], GRB.EQUAL,
            self.first_stage[('P_PV', t)]
        ) for t in range(0,self.split)]
        
        
        if self.risk_averse == True:
            # Scenario loss defined consistently with expected cost
            self.scenario_cost_exprs = [
                gp.quicksum(
                    self.variables['Q_GAS_BUY'][t] * scen.gas_prices[t]
                    + self.variables['Q_COAL_BUY'][t] * scen.coal_prices[t]
                    + (self.variables['Q_EUA_BUY'][t] - self.variables['Q_EUA_SELL'][t]) * scen.eua_prices[t]
                    for t in self.days
                )
                for scen in self.in_sample_scenarios
            ]

            # CVaR constraints
            self.cvar_constraints = [
                self.model.addLConstr(
                    self.eta[i],
                    GRB.GREATER_EQUAL,
                    self.scenario_cost_exprs[i] - self.zeta
                )
                for i in range(self.in_sample_scenarios.__len__())
            ]
        
    

    def _build_objective(self):
        scenario_weight = 1 / len(self.in_sample_scenarios)  # expected cost, uniform probs
        expected_cost = gp.quicksum(
            scenario_weight * gp.quicksum(
                self.variables['Q_GAS_BUY'][t] * scen.gas_prices[t]
                + self.variables['Q_COAL_BUY'][t] * scen.coal_prices[t]
                + (self.variables['Q_EUA_BUY'][t] - self.variables['Q_EUA_SELL'][t]) * scen.eua_prices[t]
                for t in self.days
            )
            for scen in self.in_sample_scenarios
        )

        self.model.setObjective(expected_cost, GRB.MINIMIZE)

        if self.risk_averse == True:
            alpha = self.alpha  # confidence level
            beta = self.beta    # trade-off parameter between expected cost and CVaR
            cvar_term = self.zeta + (1 / (1 - alpha)) * scenario_weight * gp.quicksum(
                self.eta[i] for i in range(self.in_sample_scenarios.__len__())
            )
            self.model.setObjective(
                (1 - beta) * expected_cost + beta * cvar_term,
                GRB.MINIMIZE
            )

    
    def _save_results(self):
        self.results.obj_val = self.model.ObjVal
        self.results.obj_vals = {
            i: gp.quicksum(
                self.variables['Q_GAS_BUY'][t].x * scen.gas_prices[t]
                + self.variables['Q_COAL_BUY'][t].x * scen.coal_prices[t]
                + (self.variables['Q_EUA_BUY'][t].x - self.variables['Q_EUA_SELL'][t].x) * scen.eua_prices[t]
                for t in self.days
            ).getValue()   # convert LinExpr to float
            for i, scen in enumerate(self.in_sample_scenarios)
        }

        self.results.var_vals = {
            (v, t): self.variables[v][t].x
            for v in self.variables
            for t in self.days
        }
    def _build_model(self):
        self.model = gp.Model(self.name)
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()
        
    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise ValueError(f"No optimal solution found for {self.model.name}")
    
    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Optimal objective value:")
        print(self.results.obj_val)
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
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        ax.legend()
        ax.grid()
        plt.show()

    def ex_post_analysis(self):
        # Get out-of-sample scenarios
        out_of_sample_scenarios = self.out_of_sample_scenarios
        
        # Evaluate the fixed first-stage decisions on out-of-sample scenarios
        ex_post_costs = []
        for scen in out_of_sample_scenarios:
            total_cost = gp.quicksum(
                self.results.var_vals[('Q_GAS_BUY', t)] * scen.gas_prices[t]
                + self.results.var_vals[('Q_COAL_BUY', t)] * scen.coal_prices[t]
                + self.results.var_vals[('Q_EUA_BUY', t)] * scen.eua_prices[t]
                for t in self.days
            ).getValue()  # convert LinExpr to float
            ex_post_costs.append(total_cost)
            
        average_ex_post_cost = sum(ex_post_costs) / len(ex_post_costs)
        print("Average Ex-Post Cost over Out-of-Sample Scenarios:")
        print(average_ex_post_cost)
        self.results.ex_post_obj_vals = ex_post_costs
        
        # Check wether constraints are satisfied in out-of-sample scenarios
        
        
        
        infeasible = [0]*len(out_of_sample_scenarios)
        infeasible_wind = 0
        infeasible_pv = 0
        
        
        for i, scen in enumerate(out_of_sample_scenarios):
            total_production = sum(
                self.results.var_vals[('P_COAL', t)] + self.results.var_vals[('P_GAS', t)]
                + scen.rhs_wind_prod[t] + scen.rhs_pv_prod[t]
                for t in self.days
            )
            if total_production < scen.rhs_demand:
                infeasible[i] = 1
            for t in self.days:
                if self.results.var_vals[('P_WIND', t)] > scen.rhs_wind_prod[t]:
                    infeasible_wind = 1
                if self.results.var_vals[('P_PV', t)] > scen.rhs_pv_prod[t]:
                    infeasible_pv = 1
            if infeasible_pv == 1 or infeasible_wind == 1:
                if infeasible[i] == 1:
                    pass
                else:
                    infeasible[i] = 1
            infeasible_pv = 0
            infeasible_wind = 0
                
            
            
        
        print(f"Number of infeasible out-of-sample scenarios: {sum(infeasible)} out of {len(out_of_sample_scenarios)}")
        



class StochasticModel_First_Stage():

    def __init__(self, input_data: InputData, name: str = "Stochastic Optimization Model", days: int = 366, n_scenario: int = 10, risk_averse: bool = False, alpha: float = 0.95, beta: float = 0.5):
        self.data = input_data
        # keep numeric day count and an iterable range for loops
        self.n_days = int(days)
        self.days = range(self.n_days)
        self.n_scenario = range(n_scenario)
        self.risk_averse = risk_averse
        self.alpha = alpha
        self.beta = beta
        self.name = name
        self.results = Expando()
        self.scenarios = self.generate_scenarios(n_scenario, sampling_method='normal')
        self._build_model()
        
        
    def generate_scenarios(self, n_scenarios: int, sampling_method: str = 'normal'):
        scenarios = []
        
        def uniform_sample(lower=0.8, upper=1.2):
            return random.uniform(lower, upper)

        def normal_sample(mu=0.0, sigma=0.1, lower=0.8, upper=1.2):

            return max(lower, min(upper, 1 + random.gauss(mu, sigma)))

        for s in range(n_scenarios):
            # Use one shared daily shock to keep prices/renewables correlated across drivers
            if sampling_method == 'normal':
                daily_shocks = [normal_sample() for _ in self.days]
            elif sampling_method == 'uniform':
                daily_shocks = [uniform_sample() for _ in self.days]
            else:
                raise ValueError(f"Unsupported sampling method: {sampling_method}")

            # Cap renewables at daily capacity, e.g., 100,000 kW
            capped_wind = [min(prod * daily_shocks[t], 24 * 150_000) for t, prod in enumerate(self.data.rhs_wind_prod)]
            capped_pv = [min(prod * daily_shocks[t], 24 * 100_000) for t, prod in enumerate(self.data.rhs_pv_prod)]
            
            scenario_data = InputData(
                variables=self.data.variables,
                gas_prices=[price * daily_shocks[t] for t, price in enumerate(self.data.gas_prices)],
                coal_prices=[price * daily_shocks[t] for t, price in enumerate(self.data.coal_prices)],
                eua_prices=[price * daily_shocks[t] for t, price in enumerate(self.data.eua_prices)],
                rhs_demand=self.data.rhs_demand,
                rhs_storage=self.data.rhs_storage,
                rhs_prod=self.data.rhs_prod,
                rhs_prod_wind=capped_wind,
                rhs_prod_pv=capped_pv,
                efficiencies=self.data.efficiencies,
                co2_per_kWh=self.data.co2_per_kWh,
                min_prod_ratio=self.data.min_prod_ratio,
                starting_storage_levels=self.data.starting_storage_levels,
                starting_eua_balance=self.data.starting_eua_balance
            )
            scenarios.append(scenario_data)
        # Split into in sample and out of sample scenarios
        split_index = int(0.8 * n_scenarios)
        self.out_of_sample_scenarios = scenarios[:split_index]
        self.in_sample_scenarios = scenarios[split_index:]
        
        return self.in_sample_scenarios, self.out_of_sample_scenarios


    def _build_variables(self):
        
        # Decsion variables
        self.variables = {
            v: [self.model.addVar(name=f"{v}_{t}") for t in self.days]
            for v in self.data.variables
        }
        # If risk averse is true add CVaR auxillary variables
        if self.risk_averse == True:
            self.zeta = self.model.addVar(name="zeta")
            # One eta per scenario captures excess loss over zeta
            self.eta = {i: self.model.addVar(name=f"eta_scen_{i}") for i in range(len(self.in_sample_scenarios))}
    
    
    def _build_constraints(self):

        # Annual demand must hold for each scenario (robust feasibility)
        self.demand = [self.model.addLConstr(
            gp.quicksum(
                self.variables['P_COAL'][t] + self.variables['P_GAS'][t]
                + self.variables['P_WIND'][t] + self.variables['P_PV'][t]
                for t in self.days
            ),
            GRB.GREATER_EQUAL,
            scen.rhs_demand
        ) for scen in self.in_sample_scenarios]

        # Storage maximum capacity per scenario
        self.storage_gas__max = [self.model.addLConstr(
            self.variables['Q_GAS_STORAGE'][t], GRB.LESS_EQUAL, scen.rhs_storage['Q_GAS_STORAGE']
        ) for t in self.days for scen in self.in_sample_scenarios]

        self.storage_coal__max = [self.model.addLConstr(
            self.variables['Q_COAL_STORAGE'][t], GRB.LESS_EQUAL, scen.rhs_storage['Q_COAL_STORAGE']
        ) for t in self.days for scen in self.in_sample_scenarios]  
        
        # Storage balance per scenario
        self.storage_gas = [self.model.addLConstr(
            self.variables['Q_GAS_BUY'][t] + self.variables['Q_GAS_STORAGE'][t-1]
            - (self.variables['P_GAS'][t] / scen.efficiencies['eta_GAS']),
            GRB.EQUAL,
            self.variables['Q_GAS_STORAGE'][t]
        ) for t in range(1, self.n_days) for scen in self.in_sample_scenarios]

        self.storage_coal = [self.model.addLConstr(
            self.variables['Q_COAL_BUY'][t] + self.variables['Q_COAL_STORAGE'][t-1]
            - (self.variables['P_COAL'][t] / scen.efficiencies['eta_COAL']),
            GRB.EQUAL,
            self.variables['Q_COAL_STORAGE'][t]
        ) for t in range(1, self.n_days) for scen in self.in_sample_scenarios]

        # Initial storage level per scenario
        self.init_storage_gas = [self.model.addLConstr(
            self.variables['Q_GAS_BUY'][0] + scen.starting_storage_levels['Q_GAS_STORAGE']
            - (self.variables['P_GAS'][0] / scen.efficiencies['eta_GAS']),
            GRB.EQUAL,
            self.variables['Q_GAS_STORAGE'][0]
        ) for scen in self.in_sample_scenarios]

        self.init_storage_coal = [self.model.addLConstr(
            self.variables['Q_COAL_BUY'][0] + scen.starting_storage_levels['Q_COAL_STORAGE']
            - (self.variables['P_COAL'][0] / scen.efficiencies['eta_COAL']),
            GRB.EQUAL,
            self.variables['Q_COAL_STORAGE'][0]
        ) for scen in self.in_sample_scenarios]

        # Final storage level per scenario (return to start)
        self.final_storage_gas = [self.model.addLConstr(
            self.variables['Q_GAS_STORAGE'][self.n_days - 1], GRB.EQUAL, scen.starting_storage_levels['Q_GAS_STORAGE']
        ) for scen in self.in_sample_scenarios]

        self.final_storage_coal = [self.model.addLConstr(
            self.variables['Q_COAL_STORAGE'][self.n_days - 1], GRB.EQUAL, scen.starting_storage_levels['Q_COAL_STORAGE']
        ) for scen in self.in_sample_scenarios]
        
        # CO2 Emission per scenario
        self.CO2_emission = [self.model.addLConstr(
            gp.quicksum(
                self.variables['P_GAS'][t] * scen.co2_per_kWh['CO2_per_kWh_GAS']
                + self.variables['P_COAL'][t] * scen.co2_per_kWh['CO2_per_kWh_COAL']
                for t in self.days
            ),
            GRB.EQUAL,
            self.variables['Q_EUA_BALANCE'][self.n_days - 1]
        ) for scen in self.in_sample_scenarios]
        
        self.EUA_balance = [self.model.addLConstr(
            self.variables['Q_EUA_BALANCE'][t-1] + self.variables['Q_EUA_BUY'][t] - self.variables['Q_EUA_SELL'][t],
            GRB.EQUAL,
            self.variables['Q_EUA_BALANCE'][t]
        ) for t in range(1, self.n_days) for scen in self.in_sample_scenarios]
        
        self.EUA_balance_init = [self.model.addLConstr(
            self.variables['Q_EUA_BALANCE'][0],
            GRB.EQUAL,
            self.data.starting_eua_balance
        ) for scen in self.in_sample_scenarios]
        
        self.EUA_max_sell = [self.model.addLConstr(
            self.variables['Q_EUA_SELL'][t], GRB.LESS_EQUAL, 1_000_000
        ) for t in self.days for scen in self.in_sample_scenarios]
        
        self.EUA_max_buy = [self.model.addLConstr(
            self.variables['Q_EUA_BUY'][t], GRB.LESS_EQUAL, 1_000_000
        ) for t in self.days for scen in self.in_sample_scenarios]
        
        # Maximum production capacity per scenario
        self.max_prod_COAL_cap = [self.model.addLConstr(
            self.variables['P_COAL'][t], GRB.LESS_EQUAL, scen.rhs_prod['P_COAL']
        ) for t in self.days for scen in self.in_sample_scenarios]

        self.max_prod_GAS_cap = [self.model.addLConstr(
            self.variables['P_GAS'][t], GRB.LESS_EQUAL, scen.rhs_prod['P_GAS']
        ) for t in self.days for scen in self.in_sample_scenarios]
        
        # Production limited by available fuel per scenario
        self.max_prod_GAS = [self.model.addLConstr(
            self.variables['P_GAS'][t], GRB.LESS_EQUAL,
            scen.efficiencies['eta_GAS'] * (self.variables['Q_GAS_STORAGE'][t-1] + self.variables['Q_GAS_BUY'][t])
        ) for t in range(1, self.n_days) for scen in self.in_sample_scenarios]

        self.max_prod_COAL = [self.model.addLConstr(
            self.variables['P_COAL'][t], GRB.LESS_EQUAL,
            scen.efficiencies['eta_COAL'] * (self.variables['Q_COAL_STORAGE'][t-1] + self.variables['Q_COAL_BUY'][t])
        ) for t in range(1, self.n_days) for scen in self.in_sample_scenarios]

        # Initial production limited by initial storage and buy
        self.max_prod_GAS_init = [self.model.addLConstr(
            self.variables['P_GAS'][0], GRB.LESS_EQUAL,
            scen.efficiencies['eta_GAS'] * (scen.starting_storage_levels['Q_GAS_STORAGE'] + self.variables['Q_GAS_BUY'][0])
        ) for scen in self.in_sample_scenarios]

        self.max_prod_COAL_init = [self.model.addLConstr(
            self.variables['P_COAL'][0], GRB.LESS_EQUAL,
            scen.efficiencies['eta_COAL'] * (scen.starting_storage_levels['Q_COAL_STORAGE'] + self.variables['Q_COAL_BUY'][0])
        ) for scen in self.in_sample_scenarios]

        self.max_prod_wind = [self.model.addLConstr(
            self.variables['P_WIND'][t], GRB.LESS_EQUAL, scen.rhs_wind_prod[t]
        ) for t in self.days for scen in self.in_sample_scenarios]
        
        self.max_prod_pv = [self.model.addLConstr(
            self.variables['P_PV'][t], GRB.LESS_EQUAL, scen.rhs_pv_prod[t]
        ) for t in self.days for scen in self.in_sample_scenarios]

        # Minimum production for coal and gas plants per scenario
        self.min_prod_COAL = [self.model.addLConstr(
            self.variables['P_COAL'][t], GRB.GREATER_EQUAL,
            scen.min_prod_ratio['min_prod_ratio_COAL'] * scen.rhs_prod['P_COAL']
        ) for t in self.days for scen in self.in_sample_scenarios]
    
        self.min_prod_GAS = [self.model.addLConstr(
            self.variables['P_GAS'][t], GRB.GREATER_EQUAL,
            scen.min_prod_ratio['min_prod_ratio_GAS'] * scen.rhs_prod['P_GAS']
        ) for t in self.days for scen in self.in_sample_scenarios]
        
        if self.risk_averse == True:
            # Scenario loss defined consistently with expected cost
            self.scenario_cost_exprs = [
                gp.quicksum(
                    self.variables['Q_GAS_BUY'][t] * scen.gas_prices[t]
                    + self.variables['Q_COAL_BUY'][t] * scen.coal_prices[t]
                    + (self.variables['Q_EUA_BUY'][t] - self.variables['Q_EUA_SELL'][t]) * scen.eua_prices[t]
                    for t in self.days
                )
                for scen in self.in_sample_scenarios
            ]

            # CVaR constraints
            self.cvar_constraints = [
                self.model.addLConstr(
                    self.eta[i],
                    GRB.GREATER_EQUAL,
                    self.scenario_cost_exprs[i] - self.zeta
                )
                for i in range(self.in_sample_scenarios.__len__())
            ]
        
            
        

        ### All variables are automtically set to be greater than or equal to zero



    def _build_objective(self):
        scenario_weight = 1 / len(self.in_sample_scenarios)  # expected cost, uniform probs
        expected_cost = gp.quicksum(
            scenario_weight * gp.quicksum(
                self.variables['Q_GAS_BUY'][t] * scen.gas_prices[t]
                + self.variables['Q_COAL_BUY'][t] * scen.coal_prices[t]
                + (self.variables['Q_EUA_BUY'][t] - self.variables['Q_EUA_SELL'][t]) * scen.eua_prices[t]
                for t in self.days
            )
            for scen in self.in_sample_scenarios
        )

        self.model.setObjective(expected_cost, GRB.MINIMIZE)

        if self.risk_averse == True:
            alpha = self.alpha  # confidence level
            beta = self.beta    # trade-off parameter between expected cost and CVaR
            cvar_term = self.zeta + (1 / (1 - alpha)) * scenario_weight * gp.quicksum(
                self.eta[i] for i in range(self.in_sample_scenarios.__len__())
            )
            self.model.setObjective(
                (1 - beta) * expected_cost + beta * cvar_term,
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
        self.results.obj_vals = {
            i: gp.quicksum(
                self.variables['Q_GAS_BUY'][t].x * scen.gas_prices[t]
                + self.variables['Q_COAL_BUY'][t].x * scen.coal_prices[t]
                + (self.variables['Q_EUA_BUY'][t].x - self.variables['Q_EUA_SELL'][t].x) * scen.eua_prices[t]
                for t in self.days
            ).getValue()   # convert LinExpr to float
            for i, scen in enumerate(self.in_sample_scenarios)
        }

        self.results.var_vals = {
            (v, t): self.variables[v][t].x
            for v in self.variables
            for t in self.days
        }

        
       
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
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        ax.legend()
        ax.grid()
        plt.show()

    def ex_post_analysis(self):
        # Get out-of-sample scenarios
        out_of_sample_scenarios = self.out_of_sample_scenarios
        
        # Evaluate the fixed first-stage decisions on out-of-sample scenarios
        ex_post_costs = []
        for scen in out_of_sample_scenarios:
            total_cost = gp.quicksum(
                self.results.var_vals[('Q_GAS_BUY', t)] * scen.gas_prices[t]
                + self.results.var_vals[('Q_COAL_BUY', t)] * scen.coal_prices[t]
                + self.results.var_vals[('Q_EUA_BUY', t)] * scen.eua_prices[t]
                for t in self.days
            ).getValue()  # convert LinExpr to float
            ex_post_costs.append(total_cost)
            
        average_ex_post_cost = sum(ex_post_costs) / len(ex_post_costs)
        print("Average Ex-Post Cost over Out-of-Sample Scenarios:")
        print(average_ex_post_cost)
        self.results.ex_post_obj_vals = ex_post_costs
        
        # Check wether constraints are satisfied in out-of-sample scenarios
        
        
        
        infeasible = [0]*len(out_of_sample_scenarios)
        infeasible_wind = 0
        infeasible_pv = 0
        
        
        for i, scen in enumerate(out_of_sample_scenarios):
            total_production = sum(
                self.results.var_vals[('P_COAL', t)] + self.results.var_vals[('P_GAS', t)]
                + scen.rhs_wind_prod[t] + scen.rhs_pv_prod[t]
                for t in self.days
            )
            if total_production < scen.rhs_demand:
                infeasible[i] = 1
            for t in self.days:
                if self.results.var_vals[('P_WIND', t)] > scen.rhs_wind_prod[t]:
                    infeasible_wind = 1
                if self.results.var_vals[('P_PV', t)] > scen.rhs_pv_prod[t]:
                    infeasible_pv = 1
            if infeasible_pv == 1 or infeasible_wind == 1:
                if infeasible[i] == 1:
                    pass
                else:
                    infeasible[i] = 1
            infeasible_pv = 0
            infeasible_wind = 0
                
            
            
        
        print(f"Number of infeasible out-of-sample scenarios: {sum(infeasible)} out of {len(out_of_sample_scenarios)}")
        
                
            
            
        
        