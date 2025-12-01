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
                 starting_storage_levels: dict[float]):
        
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

    




class StochasticModel():

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
        self.scenarios = self.generate_scenarios(n_scenario)
        self._build_model()
        
        
    def generate_scenarios(self, n_scenarios: int):
        scenarios = []

        def sample_factor(mu=0.0, sigma=0.1, lower=0.8, upper=1.2):

            return max(lower, min(upper, 1 + random.gauss(mu, sigma)))

        for s in range(n_scenarios):
            # Use one shared daily shock to keep prices/renewables correlated across drivers
            daily_shocks = [sample_factor() for _ in self.days]

            # Create a deep copy of the input data to modify
            scenario_data = InputData(
                variables=self.data.variables,
                gas_prices=[price * daily_shocks[t] for t, price in enumerate(self.data.gas_prices)],
                coal_prices=[price * daily_shocks[t] for t, price in enumerate(self.data.coal_prices)],
                eua_prices=[price * daily_shocks[t] for t, price in enumerate(self.data.eua_prices)],
                rhs_demand=self.data.rhs_demand,
                rhs_storage=self.data.rhs_storage,
                rhs_prod=self.data.rhs_prod,
                rhs_prod_wind=[prod * daily_shocks[t] for t, prod in enumerate(self.data.rhs_wind_prod)],
                rhs_prod_pv=[prod * daily_shocks[t] for t, prod in enumerate(self.data.rhs_pv_prod)],
                efficiencies=self.data.efficiencies,
                co2_per_kWh=self.data.co2_per_kWh,
                min_prod_ratio=self.data.min_prod_ratio,
                starting_storage_levels=self.data.starting_storage_levels
            )
            scenarios.append(scenario_data)
        # Split into in sample and out of sample scenarios
        split_index = int(0.8 * n_scenarios)
        self.out_of_sample_scenarios = scenarios[split_index:]
        self.in_sample_scenarios = scenarios[:split_index]
        
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
            gp.quicksum(self.variables['Q_EUA_BUY'][t] for t in self.days)
        ) for scen in self.in_sample_scenarios]
        

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
                    + self.variables['Q_EUA_BUY'][t] * scen.eua_prices[t]
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
                + self.variables['Q_EUA_BUY'][t] * scen.eua_prices[t]
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
                + self.variables['Q_EUA_BUY'][t].x * scen.eua_prices[t]
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
        
        plotting_days = range(225, 275)  # Example: days 225 to 274
        
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
            where='mid', color=colors[3]
        )
        
        ax.step(
            plotting_days,
            [self.results.var_vals[('P_PV', t)] for t in plotting_days],
            label='P_PV',
            where='mid', color = colors[5]
        )
        
        ax.set_xlabel('Day')
        ax.text(0.0, 1.07, 'Optimal Power Production Over Time', transform=ax.transAxes, fontsize=14, color='black', ha ='left')
        ax.text(0.0, 1.02, 'kWh', transform=ax.transAxes, fontsize=10, color='black', ha ='left')
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
        
        