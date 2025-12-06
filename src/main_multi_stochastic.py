import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from data_preparation import DataPreparationCSV, DataPreparationJSON 
from multi_stage_stochastic_opt_model import StochasticModel_First_Stage, StochasticModel_Second_Stage, InputData
import random
import numpy as np
from plotter import *
random.seed(42)

########### LOAD DATA AND PREPARE INPUTS FOR THE MULTI-STAGE STOCHASTIC MODEL #############

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

############ RUN MULTI-STAGE STOCHASTIC MODEL #############

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

no_stages = 2
stages = list(range(2, no_stages + 1))

n_scenarios = 100

first_stage_model= StochasticModel_First_Stage(input_data, n_scenario=n_scenarios)
first_stage_model.run()

first_stage_model._save_results()
first_stage_results = dict(first_stage_model.results.var_vals)

for stage in stages:
    
    second_stage_model = StochasticModel_Second_Stage(input_data, n_scenario=n_scenarios, stage=stage, no_stages=no_stages,
                                 first_stage_results=first_stage_results)
    second_stage_model.run()
    second_stage_model._save_results()
    first_stage_results = dict(second_stage_model.results.var_vals) # Update for next stage


second_stage_model.display_results()
second_stage_model._save_results()
second_stage_model.plot_results()
second_stage_model.ex_post_analysis()

############ PLOT DISTRIBUTIONS OF IN-SAMPLE AND OUT-OF-SAMPLE OBJECTIVE VALUES #############

obj_vals_list = list(second_stage_model.results.obj_vals.values())

plot_histogram(
    obj_vals_list,
    xlabel="Objective Value [EUR]",
    ylabel="Frequency of results across in-sample scenarios",
    title="Distribution of Objective Values Across Scenarios",
    bins=50
)
plot_histogram(
    second_stage_model.results.ex_post_obj_vals,
    xlabel="Ex Post Objective Value [EUR]",
    ylabel="Frequency of results across out-of-sample scenarios",
    title="Distribution of Ex Post Objective Values Across Out-of-Sample Scenarios",
    bins=50
)
