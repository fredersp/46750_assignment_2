import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from data_preparation import DataPreparationCSV, DataPreparationJSON 
from stochastic_opt_model import StochasticModel, InputData
from deterministic_opt_model import DeterministicModel
import random
import numpy as np
from scipy.stats import gaussian_kde
from plotter import *
import copy
random.seed(42)

########### LOAD DATA AND PREPARE INPUTS FOR THE STOCHASTIC MODEL #############

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
rhs_demand = 501_000_000 # kWh per year

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

############ RUN DETERMINISTIC MODEL #############

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
model = StochasticModel(input_data, n_scenario=5000, sampling_method='normal_with_extremes', risk_averse=True, beta=0.95)
model.run()
model.display_results()
model._save_results()
model.plot_results()

############ PLOT DISTRIBUTIONS OF IN-SAMPLE AND OUT-OF-SAMPLE OBJECTIVE VALUES #############

obj_vals_list = list(model.results.obj_vals.values())
plot_histogram(
    obj_vals_list,
    xlabel="Objective Value [EUR]",
    ylabel="Frequency of results across in-sample scenarios",
    title="Distribution of Objective Values Across Scenarios",
    bins=50
)

infeasible_count = model.ex_post_analysis()
plot_histogram(
    model.results.ex_post_obj_vals,
    xlabel="Ex Post Objective Value [EUR]",
    ylabel="Frequency of results across out-of-sample scenarios",
    title="Distribution of Ex Post Objective Values Across Out-of-Sample Scenarios",
    bins=50
)

############ EXPERIMENTS #############

############ EXPERIMENT 1: ENERGY MIX #############

tot_p_gas = sum(model.results.var_vals['P_GAS', t] for t in range(model.n_days))
tot_p_coal = sum(model.results.var_vals['P_COAL', t] for t in range(model.n_days))
tot_p_wind = sum(model.results.var_vals['P_WIND', t] for t in range(model.n_days))
tot_p_pv = sum(model.results.var_vals['P_PV', t] for t in range(model.n_days))

plot_energy_mix(tot_p_gas, tot_p_coal, tot_p_wind, tot_p_pv)

############ EXPERIMENT 2: SCENARIO INFEASIBILITY ANALYSIS #############

infeasible_count = []

n_scenarios = [100, 250, 500, 750, 1000, 1250, 1500]

for n in n_scenarios:
    model = StochasticModel(input_data, n_scenario=n, sampling_method='normal_with_extremes')
    model.run()
    infeasible = model.ex_post_analysis()
    fail_rate = infeasible / (n * 0.8)
    infeasible_count.append(fail_rate)


plot_scenario_infeasibility(n_scenarios, infeasible_count)

############ EXPERIMENT 3: IMPACT OF EUA PRICE SCENARIOS ON FUEL PURCHASES #############

EUA_scenarios = [0.5, 1.0, 1.5, 2]
total_coal_purchases_dict = {}
total_gas_purchases_dict = {}

for eua_multiplier in EUA_scenarios:
    # create a deep copy of the input data and scale EUA prices
    inp = copy.deepcopy(input_data)
    inp.eua_prices = [float(p) * eua_multiplier for p in inp.eua_prices]

    model = StochasticModel(inp, n_scenario=5000, sampling_method='normal_with_extremes', risk_averse=True, beta=0.95)
    model.run()

    # Extract total purchases per day from solved variable values
    total_coal_purchases = []
    total_gas_purchases = []
    for t in range(model.n_days):
        total_coal = float(model.results.var_vals.get(('Q_COAL_BUY', t), 0.0))
        total_gas = float(model.results.var_vals.get(('Q_GAS_BUY', t), 0.0))
        total_coal_purchases.append(total_coal)
        total_gas_purchases.append(total_gas)

    # convert to cumulative (stacked) totals so end value is total purchased
    total_coal_cumulative = list(np.cumsum(total_coal_purchases))
    total_gas_cumulative = list(np.cumsum(total_gas_purchases))

    total_coal_purchases_dict[eua_multiplier] = total_coal_cumulative
    total_gas_purchases_dict[eua_multiplier] = total_gas_cumulative

# time index for x-axis
time_index = list(range(model.n_days))

# Plot total fuel purchases over time
plot_fuel_purchases_over_time(time_index, total_coal_purchases_dict, total_gas_purchases_dict)

########## EXPERIMENT 4: RISK ANALYSIS #############

# Initialize and run the stochastic model
model = StochasticModel(input_data, n_scenario=5000, sampling_method='normal_with_extremes', risk_averse=False, beta=0.95)
model.run()
model.display_results()
model._save_results()

# Initialize and run the stochastic model
model_risk = StochasticModel(input_data, n_scenario=5000, sampling_method='normal_with_extremes', risk_averse=True, beta=0.95)
model_risk.run()
model_risk.display_results()
model_risk._save_results()

# Save objective values to list and create box plot
obj_vals_list = list(model.results.obj_vals.values())
obj_vals_list_risk = list(model_risk.results.obj_vals.values())

# Calculate VaR (95th percentile) and CVaR (mean of worst 5%)
var_neutral = np.percentile(obj_vals_list, 95)
var_risk = np.percentile(obj_vals_list_risk, 95)
cvar_neutral = np.mean([x for x in obj_vals_list if x >= var_neutral])
cvar_risk = np.mean([x for x in obj_vals_list_risk if x >= var_risk])

# Calculate expected costs
expected_cost_neutral = np.mean(obj_vals_list)
expected_cost_risk = np.mean(obj_vals_list_risk)

fig, ax = plt.subplots(figsize=(10, 6))
# Plot KDE lines instead of histograms
kde_neutral = gaussian_kde(obj_vals_list)
kde_risk = gaussian_kde(obj_vals_list_risk)
x_range = np.linspace(min(min(obj_vals_list), min(obj_vals_list_risk)), 
                    max(max(obj_vals_list), max(obj_vals_list_risk)), 200)

ax.plot(x_range, kde_neutral(x_range), linewidth=2, label='Risk-neutral', color='#B00020')
ax.plot(x_range, kde_risk(x_range), linewidth=2, label='Risk-averse (β=0.95)', color='#5C97D9')

# Shade CVaR areas
x_cvar_neutral = x_range[x_range >= var_neutral]
ax.fill_between(x_cvar_neutral, 0, kde_neutral(x_cvar_neutral), alpha=0.3, color='#B00020', label=f'CVaR (Risk-neutral): €{cvar_neutral:,.0f}', edgecolor='none')

x_cvar_risk = x_range[x_range >= var_risk]
ax.fill_between(x_cvar_risk, 0, kde_risk(x_cvar_risk), alpha=0.3, color='#5C97D9', label=f'CVaR (Risk-averse): €{cvar_risk:,.0f}', edgecolor='none')

ax.axvline(var_neutral, color='#B00020', linestyle='--', alpha=0.7, label=f'VaR (Risk-neutral): €{var_neutral:,.0f}')
ax.axvline(var_risk, color='#5C97D9', linestyle='--', alpha=0.7, label=f'VaR (Risk-averse): €{var_risk:,.0f}')

ax.axvline(expected_cost_neutral, color='#B00020', linestyle=':', linewidth=2, alpha=0.7, label=f'Expected Cost (Risk-neutral): €{expected_cost_neutral:,.0f}')
ax.axvline(expected_cost_risk, color='#5C97D9', linestyle=':', linewidth=2, alpha=0.7, label=f'Expected Cost (Risk-averse): €{expected_cost_risk:,.0f}')

ax.set_xlabel("Total annual cost [€]", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_ylim(bottom=0, top=ax.get_ylim()[1]*1.2)
ax.text(0.0, 1.07, 'Risk Analysis', transform=ax.transAxes, fontsize=18, color='black', ha='left', fontweight='bold')
ax.text(0.0, 1.03, 'Distribution of total annual cost for stochastic model: Risk-neutral vs Risk-averse', transform=ax.transAxes, fontsize=14, color='black', ha='left')  
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.show()
