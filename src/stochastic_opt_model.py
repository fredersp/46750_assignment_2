import gurobipy as gp
from gurobipy import GRB
import random










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



class ScenarioGenerator:
    
    def __init__(self, input_data: InputData, days: int = 366):
        self.data = input_data
        self.n_days = days
    
    def generate_scenarios(self, n_scenarios: int):
        scenarios = []
        
        for s in range(n_scenarios):
            # Create a deep copy of the input data to modify
            scenario_data = InputData(
                variables=self.data.variables,
                gas_prices=[price * random.uniform(0.8, 1.2) for price in self.data.gas_prices],
                coal_prices=[price * random.uniform(0.8, 1.2) for price in self.data.coal_prices],
                eua_prices=[price * random.uniform(0.8, 1.2) for price in self.data.eua_prices],
                rhs_demand=self.data.rhs_demand,
                rhs_storage=self.data.rhs_storage,
                rhs_prod=self.data.rhs_prod,
                rhs_prod_wind=[prod * random.uniform(0.8, 1.2) for prod in self.data.rhs_wind_prod],
                rhs_prod_pv=[prod * random.uniform(0.8, 1.2) for prod in self.data.rhs_pv_prod],
                efficiencies=self.data.efficiencies,
                co2_per_kWh=self.data.co2_per_kWh,
                min_prod_ratio=self.data.min_prod_ratio,
                starting_storage_levels=self.data.starting_storage_levels
            )
            scenarios.append(scenario_data)
        
        
        return scenarios
        
        
    
    




class StochasticModel():

    def __init__(self, input_data: InputData, name: str = "Stochastic Optimization Model", days: int = 366, n_scenario: int = 10):
        self.data = input_data
        # keep numeric day count and an iterable range for loops
        self.n_days = int(days)
        self.days = range(self.n_days)
        self.n_scenario = range(n_scenario)
        self.name = name
        self.results = Expando()
        self.scenarios = self.generate_scenarios(n_scenario)
        self._build_model()
        
        
    def generate_scenarios(self, n_scenarios: int):
        scenarios = []
        
        for s in range(n_scenarios):
            # Create a deep copy of the input data to modify
            scenario_data = InputData(
                variables=self.data.variables,
                gas_prices=[price * random.uniform(0.9, 1.1) for price in self.data.gas_prices],
                coal_prices=[price * random.uniform(0.9, 1.1) for price in self.data.coal_prices],
                eua_prices=[price * random.uniform(0.9, 1.1) for price in self.data.eua_prices],
                rhs_demand=self.data.rhs_demand,
                rhs_storage=self.data.rhs_storage,
                rhs_prod=self.data.rhs_prod,
                rhs_prod_wind=[prod * random.uniform(0.9, 1.1) for prod in self.data.rhs_wind_prod],
                rhs_prod_pv=[prod * random.uniform(0.9, 1.1) for prod in self.data.rhs_pv_prod],
                efficiencies=self.data.efficiencies,
                co2_per_kWh=self.data.co2_per_kWh,
                min_prod_ratio=self.data.min_prod_ratio,
                starting_storage_levels=self.data.starting_storage_levels
            )
            scenarios.append(scenario_data)
        
        
        return scenarios

    def _build_variables(self):
        # variables[var][t][s] -> Gurobi var
        self.variables = {
            v: [
                [self.model.addVar(name=f"{v}_{t}_{s}") for s in self.n_scenario]
                for t in self.days
                ]
            for v in self.data.variables  # or self.scenarios[0].variables
        }

    def _build_constraints(self):

        # Annual demand constraint
        self.demand = [self.model.addLConstr(
            gp.quicksum(self.variables['P_COAL'][t][s] + self.variables['P_GAS'][t][s] + self.variables['P_WIND'][t][s] + self.variables['P_PV'][t][s] for t in self.days),
                         GRB.GREATER_EQUAL, self.scenarios[s].rhs_demand)
            for s in self.n_scenario
        ]
        # Storage maximum capacity constraints
        self.storage_gas__max = [self.model.addLConstr(
            self.variables['Q_GAS_STORAGE'][t][s], GRB.LESS_EQUAL, self.scenarios[s].rhs_storage['Q_GAS_STORAGE']
            )   
            for t in self.days 
            for s in self.n_scenario
        ]
        
        self.storage_coal__max = [self.model.addLConstr(
            self.variables['Q_COAL_STORAGE'][t][s], GRB.LESS_EQUAL, self.scenarios[s].rhs_storage['Q_COAL_STORAGE']
            )
            for t in self.days
            for s in self.n_scenario
        ]
        
        # Storage balance constraints
        self.storage_gas = [self.model.addLConstr(
             self.variables['Q_GAS_BUY'][t][s] + self.variables['Q_GAS_STORAGE'][t-1][s] - (self.variables['P_GAS'][t][s] / self.scenarios[s].efficiencies['eta_GAS']), GRB.EQUAL, self.variables['Q_GAS_STORAGE'][t][s]
            )
            for t in range(1, self.n_days)
            for s in self.n_scenario
        ]
                        
        
        self.storage_coal = [self.model.addLConstr(
             self.variables['Q_COAL_BUY'][t][s] + self.variables['Q_COAL_STORAGE'][t-1][s] - (self.variables['P_COAL'][t][s] / self.scenarios[s].efficiencies['eta_COAL']), GRB.EQUAL, self.variables['Q_COAL_STORAGE'][t][s] 
            )
            for t in range(1, self.n_days)
            for s in self.n_scenario
        ]
        
        # Initial storage level constraints
        self.init_storage_gas = [self.model.addLConstr(
            self.variables['Q_GAS_BUY'][0][s] + self.scenarios[s].starting_storage_levels['Q_GAS_STORAGE'] - (self.variables['P_GAS'][0][s] / self.scenarios[s].efficiencies['eta_GAS']), GRB.EQUAL, self.variables['Q_GAS_STORAGE'][0][s]
        ) for s in self.n_scenario
        ]
        
        self.init_storage_coal = [self.model.addLConstr(
            self.variables['Q_COAL_BUY'][0][s] + self.scenarios[s].starting_storage_levels['Q_COAL_STORAGE'] - (self.variables['P_COAL'][0][s] / self.scenarios[s].efficiencies['eta_COAL']), GRB.EQUAL, self.variables['Q_COAL_STORAGE'][0][s]
        ) for s in self.n_scenario
        ]
        
        # Final storage level constraints
        self.final_storage_gas = [self.model.addLConstr(
            self.variables['Q_GAS_STORAGE'][self.n_days - 1][s], GRB.EQUAL, self.scenarios[s].starting_storage_levels['Q_GAS_STORAGE']
        ) for s in self.n_scenario
        ]
        self.final_storage_coal = [self.model.addLConstr(
            self.variables['Q_COAL_STORAGE'][self.n_days - 1][s], GRB.EQUAL, self.scenarios[s].starting_storage_levels['Q_COAL_STORAGE']
        ) for s in self.n_scenario
        ]
        
        # CO2 Emission
        self.CO2_emission = [self.model.addLConstr(
            gp.quicksum(self.variables['P_GAS'][t][s] * self.scenarios[s].co2_per_kWh['CO2_per_kWh_GAS'] + self.variables['P_COAL'][t][s] * self.scenarios[s].co2_per_kWh['CO2_per_kWh_COAL'] for t in self.days), GRB.EQUAL, gp.quicksum(self.variables['Q_EUA'][t][s] for t in self.days)
        ) for s in self.n_scenario
        ]
        
        # Maximum production constraints
        self.max_prod_COAL_cap = [ self.model.addLConstr(
            self.variables['P_COAL'][t][s], GRB.LESS_EQUAL, self.scenarios[s].rhs_prod['P_COAL']
        )
        for t in self.days
        for s in self.n_scenario
        ]
        
        self.max_prod_GAS_cap = [ self.model.addLConstr(
            self.variables['P_GAS'][t][s], GRB.LESS_EQUAL, self.scenarios[s].rhs_prod['P_GAS']
        )
        for t in self.days
        for s in self.n_scenario
        ]
        
        # Maximum production constrainted by resource availability
        self.max_prod_GAS = [ self.model.addLConstr(
            self.variables['P_GAS'][t][s], GRB.LESS_EQUAL, self.scenarios[s].efficiencies['eta_GAS'] * (self.variables['Q_GAS_STORAGE'][t-1][s] + self.variables['Q_GAS_BUY'][t][s])
        ) 
            for t in range(1, self.n_days)
            for s in self.n_scenario
        ]
        
        self.max_prod_COAL = [ self.model.addLConstr(
            self.variables['P_COAL'][t][s], GRB.LESS_EQUAL, self.scenarios[s].efficiencies['eta_COAL'] * (self.variables['Q_COAL_STORAGE'][t-1][s] + self.variables['Q_COAL_BUY'][t][s])
        ) 
            for t in range(1, self.n_days)
            for s in self.n_scenario
        ]
        
        # Initial maximum production constrainted by resource availability
        self.max_prod_GAS_init = [self.model.addLConstr(
            self.variables['P_GAS'][0][s], GRB.LESS_EQUAL, self.scenarios[s].efficiencies['eta_GAS'] * (self.scenarios[s].starting_storage_levels['Q_GAS_STORAGE'] + self.variables['Q_GAS_BUY'][0][s])
        ) for s in self.n_scenario
        ]
        
        self.max_prod_COAL_init = [self.model.addLConstr(
            self.variables['P_COAL'][0][s], GRB.LESS_EQUAL, self.scenarios[s].efficiencies['eta_COAL'] * (self.scenarios[s].starting_storage_levels['Q_COAL_STORAGE'] + self.variables['Q_COAL_BUY'][0][s])
        ) for s in self.n_scenario
        ]
                                   
        
        self.max_prod_wind = [ self.model.addLConstr(
            self.variables['P_WIND'][t][s], GRB.LESS_EQUAL, self.scenarios[s].rhs_wind_prod[t]
        )
        for t in self.days
        for s in self.n_scenario
        ]  
        
        self.max_prod_pv = [ self.model.addLConstr(
            self.variables['P_PV'][t][s], GRB.LESS_EQUAL, self.scenarios[s].rhs_pv_prod[t]
        )
        for t in self.days
        for s in self.n_scenario
        ]
        
        
        # Minimum production for coal and gas plants
        self.min_prod_COAL = [ self.model.addLConstr(
            self.variables['P_COAL'][t][s], GRB.GREATER_EQUAL, self.scenarios[s].min_prod_ratio['min_prod_ratio_COAL'] * self.scenarios[s].rhs_prod['P_COAL']
        )
        for t in self.days
        for s in self.n_scenario
        ]
        
        self.min_prod_GAS = [ self.model.addLConstr(
            self.variables['P_GAS'][t][s], GRB.GREATER_EQUAL, self.scenarios[s].min_prod_ratio['min_prod_ratio_GAS'] * self.scenarios[s].rhs_prod['P_GAS']
        )
        for t in self.days
        for s in self.n_scenario
        ]
        

        ### All variables are automtically set to be greater than or equal to zero



    def _build_objective(self):
        scenario_weight = 1 / len(self.n_scenario)  # if you want expected cost
        self.model.setObjective(
            gp.quicksum(
                scenario_weight * gp.quicksum(
                    self.variables['Q_GAS_BUY'][t][s] * self.scenarios[s].gas_prices[t]
                    + self.variables['Q_COAL_BUY'][t][s] * self.scenarios[s].coal_prices[t]
                    + self.variables['Q_EUA'][t][s] * self.scenarios[s].eua_prices[t]
                    for t in self.days
                )
                for s in self.n_scenario
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
        self.results.obj_vals = {
            s: gp.quicksum(
                self.variables['Q_GAS_BUY'][t][s].x * self.scenarios[s].gas_prices[t]
                + self.variables['Q_COAL_BUY'][t][s].x * self.scenarios[s].coal_prices[t]
                + self.variables['Q_EUA'][t][s].x * self.scenarios[s].eua_prices[t]
                for t in self.days
            ).getValue()   # convert LinExpr to float
            for s in self.n_scenario
    }
        self.results.var_vals = {
        (v, t, s): self.variables[v][t][s].x
        for v in self.variables
        for t in self.days
        for s in self.n_scenario
    }

        
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
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        
        plt.plot(
            self.days,
            [self.results.var_vals[('P_GAS', t)] for t in self.days],
            label='P_GAS'
        )
        
        plt.plot(
            self.days,
            [self.results.var_vals[('P_COAL', t)] for t in self.days],
            label='P_COAL'
        )
        
        plt.plot(
            self.days,
            [self.results.var_vals[('P_WIND', t)] for t in self.days],
            label='P_WIND'
        )
        
        plt.plot(
            self.days,
            [self.results.var_vals[('P_PV', t)] for t in self.days],
            label='P_PV'
        )
        
        plt.xlabel('Day')
        plt.ylabel('Power Production (kWh)')
        plt.title('Optimal Power Production Over Time')
        plt.legend()
        plt.grid()
        plt.show()
        
        plt.figure(figsize=(12, 8))
        plt.bar(
            self.days,
            [self.results.var_vals[('Q_GAS_STORAGE', t)] for t in self.days],
            label='Q_GAS_STORAGE'
        )
        plt.bar(
            self.days,
            [self.results.var_vals[('Q_COAL_STORAGE', t)] for t in self.days],
            label='Q_COAL_STORAGE'
        )
        plt.xlabel('Day')
        plt.ylabel('Storage Level (kWh)')
        plt.title('Optimal Storage Levels Over Time')
        plt.legend()
        plt.grid()
        plt.show()
        
        plt.figure(figsize=(12, 8))
        plt.plot(
            self.days,
            [self.results.var_vals[('Q_GAS_BUY', t)] for t in self.days],
            label='Q_GAS_BUY'
        )
        plt.plot(
            self.days,
            [self.results.var_vals[('Q_COAL_BUY', t)] for t in self.days],
            label='Q_COAL_BUY'
        )
        plt.xlabel('Day')
        plt.ylabel('Fuel Purchased (kWh)')
        plt.title('Optimal Fuel Purchases Over Time')
        plt.legend()
        plt.grid()
        plt.show()
        
        plt.figure(figsize=(12, 8))
        plt.bar(
            self.days,
            [self.results.var_vals[('Q_EUA', t)] for t in self.days],
            label='Q_EUA'
        )
        plt.xlabel('Day')
        plt.ylabel('EUA Purchased (kgCO2eq)')
        plt.title('Optimal EUA Purchases Over Time')
        plt.legend()
        plt.grid()
        plt.show()