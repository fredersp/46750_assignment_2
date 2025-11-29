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

    def __init__(self, input_data: InputData, name: str = "Stochastic Optimization Model", days: int = 366, n_scenario: int = 10, n_stages: int = 2):
        self.data = input_data
        # keep numeric day count and an iterable range for loops
        self.n_days = int(days)
        self.days = range(self.n_days)
        self.n_scenario = range(n_scenario)
        self.k = [k*self.n_days/n_stages for k in range(1,n_stages)]# length of each stage
        self.stages = n_stages
        self.name = name
        self.results = Expando()
        
        self.tree = self.build_scenario_tree(n_scenario, n_stages)
        #self.scenarios = self.generate_scenarios(n_scenario)
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
    
    def build_scenario_tree(self, n_scenarios: int, n_stages: int):
        """
        Build a full scenario tree with n_stages, each node having n_scenarios children.
        Node format:
        {
            "stage": int,            # 0..n_stages-1
            "path": tuple[int, ...], # sequence of branch ids from root
            "parent": tuple|None,    # parent path
            "prob": float,           # cumulative probability
            "data": InputData        # scenario data for this node
          }
        """
        if n_stages < 1:
            return []

        tree = []

        def add_stage_nodes(stage: int, parent_path: tuple[int, ...], parent_prob: float):
            # sample new scenarios for this stage
            scenarios = self.generate_scenarios(n_scenarios)
            child_prob = parent_prob / n_scenarios
            for i, scen in enumerate(scenarios):
                path = parent_path + (i,)
                tree.append({
                    "stage": stage,
                    "path": path,
                    "parent": parent_path if parent_path else None,
                    "prob": child_prob,
                    "data": scen,
                })
                # recurse if not at last stage
                if stage + 1 < n_stages:
                    add_stage_nodes(stage + 1, path, child_prob)

        # root has no data; children of root are stage 0 nodes
        add_stage_nodes(stage=0, parent_path=tuple(), parent_prob=1.0)
        
        return tree

        
        
        

    def _build_variables(self):
        """
        Creates vars[var][t][node_id], where node_id indexes self.tree (output of build_scenario_tree).
        node_id carries (stage, path, prob, data).
        """
        self.variables = {
            v: [
                [self.model.addVar(name=f"{v}_t{t}_n{n}") for n, _ in enumerate(self.tree)]
                for t in self.days
            ]
            for v in self.data.variables
        }


    def _build_constraints(self):

        # Annual demand constraint
        self.demand = [self.model.addLConstr(
            gp.quicksum(self.variables['P_COAL'][t][n] + self.variables['P_GAS'][t][n] + self.variables['P_WIND'][t][n] + self.variables['P_PV'][t][n] for t in self.days),
                         GRB.GREATER_EQUAL, node["data"].rhs_demand)
            for n, node in enumerate(self.tree)
        ]
        # Storage maximum capacity constraints
        self.storage_gas__max = [self.model.addLConstr(
            self.variables['Q_GAS_STORAGE'][t][n], GRB.LESS_EQUAL, node["data"].rhs_storage['Q_GAS_STORAGE']
            )   
            for t in self.days 
            for n, node in enumerate(self.tree)
        ]
        
        self.storage_coal__max = [self.model.addLConstr(
            self.variables['Q_COAL_STORAGE'][t][n], GRB.LESS_EQUAL, node['data'].rhs_storage['Q_COAL_STORAGE']
            )
            for t in self.days
            for n, node in enumerate(self.tree)
        ]
        
        # Storage balance constraints
        self.storage_gas = [self.model.addLConstr(
             self.variables['Q_GAS_BUY'][t][n] + self.variables['Q_GAS_STORAGE'][t-1][n] - (self.variables['P_GAS'][t][n] / node['data'].efficiencies['eta_GAS']), GRB.EQUAL, self.variables['Q_GAS_STORAGE'][t][n]
            )
            for t in range(1, self.n_days)
            for n, node in enumerate(self.tree)
        ]
                        
        
        self.storage_coal = [self.model.addLConstr(
             self.variables['Q_COAL_BUY'][t][n] + self.variables['Q_COAL_STORAGE'][t-1][n] - (self.variables['P_COAL'][t][n] / node['data'].efficiencies['eta_COAL']), GRB.EQUAL, self.variables['Q_COAL_STORAGE'][t][n] 
            )
            for t in range(1, self.n_days)
            for n, node in enumerate(self.tree)
        ]
        
        # Initial storage level constraints
        self.init_storage_gas = [self.model.addLConstr(
            self.variables['Q_GAS_BUY'][0][n] + node['data'].starting_storage_levels['Q_GAS_STORAGE'] - (self.variables['P_GAS'][0][n] / node['data'].efficiencies['eta_GAS']), GRB.EQUAL, self.variables['Q_GAS_STORAGE'][0][n]
        ) for n, node in enumerate(self.tree)
        ]
        
        self.init_storage_coal = [self.model.addLConstr(
            self.variables['Q_COAL_BUY'][0][n] + node['data'].starting_storage_levels['Q_COAL_STORAGE'] - (self.variables['P_COAL'][0][n] / node['data'].efficiencies['eta_COAL']), GRB.EQUAL, self.variables['Q_COAL_STORAGE'][0][n]
        ) for n, node in enumerate(self.tree)
        ]
        
        # Final storage level constraints
        self.final_storage_gas = [self.model.addLConstr(
            self.variables['Q_GAS_STORAGE'][self.n_days - 1][n], GRB.EQUAL, node['data'].starting_storage_levels['Q_GAS_STORAGE']
        ) for n, node in enumerate(self.tree)
        ]
        
        self.final_storage_coal = [self.model.addLConstr(
            self.variables['Q_COAL_STORAGE'][self.n_days - 1][n], GRB.EQUAL, node['data'].starting_storage_levels['Q_COAL_STORAGE']
        ) for n, node in enumerate(self.tree)
        ]
        
        # CO2 Emission
        self.CO2_emission = [self.model.addLConstr(
            gp.quicksum(self.variables['P_GAS'][t][n] * node['data'].co2_per_kWh['CO2_per_kWh_GAS'] + self.variables['P_COAL'][t][n] * node['data'].co2_per_kWh['CO2_per_kWh_COAL'] for t in self.days), GRB.EQUAL, gp.quicksum(self.variables['Q_EUA'][t][n] for t in self.days)
        ) for n, node in enumerate(self.tree)
        ]
        
        # Maximum production constraints
        self.max_prod_COAL_cap = [ self.model.addLConstr(
            self.variables['P_COAL'][t][n], GRB.LESS_EQUAL, node['data'].rhs_prod['P_COAL']
        )
        for t in self.days
        for n, node in enumerate(self.tree)
        ]
        
        self.max_prod_GAS_cap = [ self.model.addLConstr(
            self.variables['P_GAS'][t][n], GRB.LESS_EQUAL, node['data'].rhs_prod['P_GAS']
        )
        for t in self.days
        for n, node in enumerate(self.tree)
        ]
        
        # Maximum production constrainted by resource availability
        self.max_prod_GAS = [ self.model.addLConstr(
            self.variables['P_GAS'][t][n], GRB.LESS_EQUAL, node['data'].efficiencies['eta_GAS'] * (self.variables['Q_GAS_STORAGE'][t-1][n] + self.variables['Q_GAS_BUY'][t][n])
        ) 
            for t in range(1, self.n_days)
            for n, node in enumerate(self.tree)
        ]
        
        self.max_prod_COAL = [ self.model.addLConstr(
            self.variables['P_COAL'][t][n], GRB.LESS_EQUAL, node['data'].efficiencies['eta_COAL'] * (self.variables['Q_COAL_STORAGE'][t-1][n] + self.variables['Q_COAL_BUY'][t][n])
        ) 
            for t in range(1, self.n_days)
            for n, node in enumerate(self.tree)
        ]
        
        # Initial maximum production constrainted by resource availability
        self.max_prod_GAS_init = [self.model.addLConstr(
            self.variables['P_GAS'][0][n], GRB.LESS_EQUAL, node['data'].efficiencies['eta_GAS'] * (node['data'].starting_storage_levels['Q_GAS_STORAGE'] + self.variables['Q_GAS_BUY'][0][n])
        ) for n, node in enumerate(self.tree)
        ]
        
        self.max_prod_COAL_init = [self.model.addLConstr(
            self.variables['P_COAL'][0][n], GRB.LESS_EQUAL, node['data'].efficiencies['eta_COAL'] * (node['data'].starting_storage_levels['Q_COAL_STORAGE'] + self.variables['Q_COAL_BUY'][0][n])
        ) for n, node in enumerate(self.tree)
        ]
                                   
        
        self.max_prod_wind = [ self.model.addLConstr(
            self.variables['P_WIND'][t][n], GRB.LESS_EQUAL, node['data'].rhs_wind_prod[t]
        )
        for t in self.days
        for n, node in enumerate(self.tree)
        ]  
        
        self.max_prod_pv = [ self.model.addLConstr(
            self.variables['P_PV'][t][n], GRB.LESS_EQUAL, node['data'].rhs_pv_prod[t]
        )
        for t in self.days
        for n, node in enumerate(self.tree)
        ]
        
        
        # Minimum production for coal and gas plants
        self.min_prod_COAL = [ self.model.addLConstr(
            self.variables['P_COAL'][t][n], GRB.GREATER_EQUAL, node['data'].min_prod_ratio['min_prod_ratio_COAL'] * node['data'].rhs_prod['P_COAL']
        )
        for t in self.days
        for n, node in enumerate(self.tree)
        ]
        
        self.min_prod_GAS = [ self.model.addLConstr(
            self.variables['P_GAS'][t][n], GRB.GREATER_EQUAL, node['data'].min_prod_ratio['min_prod_ratio_GAS'] * node['data'].rhs_prod['P_GAS']
        )
        for t in self.days
        for n, node in enumerate(self.tree)
        ]
        
        #Non-anticipative constraints
        
        

        ### All variables are automtically set to be greater than or equal to zero



    def _build_objective(self):
        #prob = [node["prob"] for node in self.tree]
        self.model.setObjective(
            gp.quicksum(
                1/(len(self.n_scenario)**self.stages) * gp.quicksum(
                    self.variables['Q_GAS_BUY'][t][n] * node['data'].gas_prices[t]
                    + self.variables['Q_COAL_BUY'][t][n] * node['data'].coal_prices[t]
                    + self.variables['Q_EUA'][t][n] * node['data'].eua_prices[t]
                    for t in self.days
                )
                for n, node in enumerate(self.tree)
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
    #     self.results.obj_vals = {
    #         s: gp.quicksum(
    #             self.variables['Q_GAS_BUY'][t][s].x * self.scenarios[s].gas_prices[t]
    #             + self.variables['Q_COAL_BUY'][t][s].x * self.scenarios[s].coal_prices[t]
    #             + self.variables['Q_EUA'][t][s].x * self.scenarios[s].eua_prices[t]
    #             for t in self.days
    #         ).getValue()   # convert LinExpr to float
    #         for s in self.n_scenario
    # }
        self.results.var_vals = {
        (v, t, n): self.variables[v][t][n].x
        for v in self.variables
        for t in self.days
        for n, node in enumerate(self.tree)
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