import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from data_preparation import DataPreparationCSV, DataPreparationJSON 
from stochastic_opt_model import StochasticModel, InputData
import random
import numpy as np
from plotter import *
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



variables = df_app['DER_id'].tolist() + df_stor['storage_id'].tolist() + ['Q_COAL_BUY', 'Q_GAS_BUY', 'Q_EUA_BUY', 'Q_EUA_SELL', 'Q_EUA_BALANCE']
coal_prices = df_t['Coal_Price[EUR/KWh]'].tolist()
gas_prices = df_t['Gas_Price[EUR/KWh]'].tolist()
eua_prices = df_t['ETS_Price[EUR/kgCO2eq]'].tolist()
rhs_demand = 501_000_000 # DO NOT CHANGE

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

starting_eua_balance = 0.0  # Assuming starting EUA balance is zero

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
    starting_storage_levels,
    starting_eua_balance
)

# Initialize and run the stochastic model
model = StochasticModel(input_data, n_scenario=5000, sampling_method='normal_with_extremes', risk_averse=True, beta=0.95)
model.run()
model.display_results()
model._save_results()
model.plot_results()


# Save objective values to list and create box plot
obj_vals_list = list(model.results.obj_vals.values())
plot_histogram(
    obj_vals_list,
    xlabel="Objective Value [EUR]",
    ylabel="Frequency of results across in-sample scenarios",
    title="Distribution of Objective Values Across Scenarios",
    bins=50
)

# Perform ex post analysis
infeasible_count = model.ex_post_analysis()
# Plot histogram of ex-post objective values
plot_histogram(
    model.results.ex_post_obj_vals,
    xlabel="Ex Post Objective Value [EUR]",
    ylabel="Frequency of results across out-of-sample scenarios",
    title="Distribution of Ex Post Objective Values Across Out-of-Sample Scenarios",
    bins=50
)


# tot_p_gas = sum(model.results.var_vals['P_GAS', t] for t in range(model.n_days))
# tot_p_coal = sum(model.results.var_vals['P_COAL', t] for t in range(model.n_days))
# tot_p_wind = sum(model.results.var_vals['P_WIND', t] for t in range(model.n_days))
# tot_p_pv = sum(model.results.var_vals['P_PV', t] for t in range(model.n_days))

# plot_energy_mix(tot_p_gas, tot_p_coal, tot_p_wind, tot_p_pv)


# betavalues = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
# gas_prod = []
# coal_prod = []
# gas_storage_dual = []
# coal_storage_dual = []

# for beta in betavalues:
#     model = StochasticModel(input_data, n_scenario=5000, risk_averse=True, beta=beta, sampling_method='normal_with_extremes')
#     model.run()
#     gas_prod.append(sum(model.results.var_vals['P_GAS', t] for t in range(model.n_days))/(8760*27500))
#     coal_prod.append(sum(model.results.var_vals['P_COAL', t] for t in range(model.n_days))/(8760*35000))
#     gas_storage_dual.append(np.mean([model.results.dual_vals['storage_gas_max'] ]))
#     coal_storage_dual.append(np.mean([model.results.dual_vals['storage_coal_max'] ]))

# # Plot gas and coal production vs beta values
# # plot_gas_vs_coal_production(betavalues, gas_prod, coal_prod)
# # # NOTE: The more risk averse (higher beta), the less gas power production, and more coal power production. As overall trend, with some fluctuations.


# plot_storage_dual_vs_beta(betavalues, gas_storage_dual, coal_storage_dual)
# NOTE: The more risk averse (higher beta), the higher the dual values for both gas and coal storage capacity constraints. Indicating increased value of storage capacity under risk aversion.



# TODO: Compare Energy Mix in stocha model vs deterministic model
# TODO: Zoom in on 260-366 days to look at the different models act how they act