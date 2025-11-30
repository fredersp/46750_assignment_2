import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from data_preparation import DataPreparationCSV, DataPreparationJSON 
from multistage_stochastic_opt_model import StochasticModel, InputData
import random
random.seed(42)


# Load data files and prepare the dataset for timeseries
data = DataPreparationCSV(datetime(2024,1,1), datetime(2024,12,31),
        coal_file_name="CoalDailyPrices.csv",
        ets_file_name="ETSDailyPrices.csv",
        gas_file_name="GasDailyBalancingPrice.csv",
        wind_file_name="wind_power_prod.csv",
        pv_file_name="pv_power_prod.csv"
    )

df_t = data.build()

params = DataPreparationJSON("appliance_params.json", "storage_params.json")
df_app = params.appliance_data_preparation()
df_stor = params.storage_data_preparation()



variables = df_app['DER_id'].tolist() + df_stor['storage_id'].tolist() + ['Q_COAL_BUY', 'Q_GAS_BUY', 'Q_EUA']
coal_prices = df_t['Coal_Price[EUR/KWh]'].tolist()
gas_prices = df_t['Gas_Price[EUR/KWh]'].tolist()
eua_prices = df_t['ETS_Price[EUR/kgCO2eq]'].tolist()
rhs_demand = 501_000_000 # DO NOT CHANGE MAYBE
# Dictionary to map coal and gas to their respective storage capacities
rhs_storage = {
    'Q_COAL_STORAGE': df_stor.loc[df_stor['storage_id'] == 'Q_COAL_STORAGE', 'capacity_kWh_fuel'].values[0],
    'Q_GAS_STORAGE': df_stor.loc[df_stor['storage_id'] == 'Q_GAS_STORAGE', 'capacity_kWh_fuel'].values[0]
}

rhs_prod = {
    'P_COAL': df_app.loc[df_app['DER_id'] == 'P_COAL', 'max_power_kW'].values[0]*24,
    'P_GAS': df_app.loc[df_app['DER_id'] == 'P_GAS', 'max_power_kW'].values[0]*24
}
rhs_prod_wind = df_t['Wind_Prod[KWh]'].tolist()
rhs_prod_pv = df_t['PV_Prod[KWh]'].tolist()

efficiencies = {
    'eta_COAL': df_app.loc[df_app['DER_id'] == 'P_COAL', 'efficiency'].values[0],
    'eta_GAS': df_app.loc[df_app['DER_id'] == 'P_GAS', 'efficiency'].values[0]
}

co2_per_kWh = {
    'CO2_per_kWh_COAL': df_app.loc[df_app['DER_id'] == 'P_COAL', 'CO2_per_kWh'].values[0],
    'CO2_per_kWh_GAS': df_app.loc[df_app['DER_id'] == 'P_GAS', 'CO2_per_kWh'].values[0]
}

min_prod_ratio = {
    'min_prod_ratio_COAL': df_app.loc[df_app['DER_id'] == 'P_COAL', 'min_power_ratio'].values[0],
    'min_prod_ratio_GAS': df_app.loc[df_app['DER_id'] == 'P_GAS', 'min_power_ratio'].values[0]
}

starting_storage_levels = {
    'Q_COAL_STORAGE': df_stor.loc[df_stor['storage_id'] == 'Q_COAL_STORAGE', 'starting_level_kWh_fuel'].values[0],
    'Q_GAS_STORAGE': df_stor.loc[df_stor['storage_id'] == 'Q_GAS_STORAGE', 'starting_level_kWh_fuel'].values[0]
}

input_data = InputData(
    variables,
    gas_prices,
    coal_prices,
    eua_prices,
    rhs_demand,
    rhs_storage,
    rhs_prod,
    rhs_prod_wind,
    rhs_prod_pv,
    efficiencies,
    co2_per_kWh,
    min_prod_ratio,
    starting_storage_levels
)


model = StochasticModel(input_data)
model.run()
model.display_results()
model._save_results()

# Save objective values to list and create box plot
obj_vals_list = list(model.results.obj_vals.values())

plt.figure(figsize=(8, 6))
plt.boxplot(obj_vals_list)
plt.ylabel("Objective Value [EUR]")
plt.title("Distribution of Scenario Costs")
plt.tight_layout()
plt.show()

# plot the distrubtion of the objective values
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile

plt.figure(figsize=(10, 6))
plt.hist(obj_vals_list, bins=30, color="#4a90e2", edgecolor='black', alpha=0.7)
plt.xlabel("Objective Value [EUR]")
plt.ylabel("Frequency")
plt.title("Distribution of Objective Values Across Scenarios")
# Calculate CVaR at 90%
alpha = 90
cvar_90 = np.mean([cost for cost in obj_vals_list if cost >= scoreatpercentile(obj_vals_list, alpha)])
plt.axvline(cvar_90, color='red', linestyle='dashed', linewidth=2, label=f'CVaR at {alpha}%: {cvar_90:.2f} EUR')
plt.legend()
plt.tight_layout()
plt.show()



# Please create a plot that shows the distribution of the objective values across all scenarios and compute the CVaR at 90%
# import matplotlib.pyplot as plt
# scenario_ids = list(model.results.obj_vals.keys())
# scenario_costs = [model.results.obj_vals[s] for s in scenario_ids]





# plt.bar(scenario_ids, scenario_costs, color="#4a90e2")
# plt.xlabel("Scenario")
# plt.ylabel("Objective value")
# plt.title("Scenario costs")
# plt.tight_layout()
# plt.show()

# Plot histogram for each of the scenarios objective values


# Plot gas prices for each scenario first 25 days
# import matplotlib.pyplot as plt

# for i, scenario in enumerate(scenarios):
#     plt.plot(scenarios[i].eua_prices[:25], label=f'Scenario {i+1}')
# plt.xlabel('Days')
# plt.ylabel('Gas Prices [EUR/KWh]')
# plt.title('Gas Price Scenarios')
# plt.legend()
# plt.show()

# TODO: Implement ex post analysis 
# TODO: Implement risk analysis (e.g., VaR, CVaR)
# TODO: Look at scenario generation methods
# TODO: Implement multi-stage stochastic problem
# TODO: Introduce Non-Anticipativity constraints
# TODO: Introduce selling EUA allowances