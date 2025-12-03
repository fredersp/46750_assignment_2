import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from data_preparation import DataPreparationCSV, DataPreparationJSON 
from stochastic_opt_model import StochasticModel, InputData
import random
import numpy as np
from plotter import plot_histogram, plot_time_series
random.seed(42)




def first_stage_optimization(input_data: InputData, n_scenarios: int):
    """Run first-stage optimization over the full year with given number of scenarios."""
    model = StochasticModel(input_data, n_scenario=n_scenarios)
    model.run()
    
    return model

def second_stage_optimization(n_scenarios: int, stages: int, first_stage_mode)
    
    

def multi_stage_optimization(n_scenarios: int, input_data: InputData, stages: int):
    
    first_stage_optimization(input_data: InputData, days: int):
    realized_result()
    
    for stage in range(stages):
        second_stage_optimization(first_stage_optimization, realized_result, stage)
        



    
    

    
def realized_result():
    #model = first_stage_optimization(input_data, n_scenarios=100)
    
    # Construct realized data for the first half-year (days 0-181)
    realized_first_stage = {
        'gas_prices': input_data.gas_prices[:182],
        'coal_prices': input_data.coal_prices[:182],
        'eua_prices': input_data.eua_prices[:182],
        'rhs_wind_prod': input_data.rhs_wind_prod[:182],
        'rhs_pv_prod': input_data.rhs_pv_prod[:182]
    }
    
    # Calculate results based on realized data
    
    days = len(realized_first_half['gas_prices'])
    
    # Optimal decisions for the first stage
    implemented_decisions = {
        'P_GAS': [model.results.var_vals[('P_GAS', t)] for t in range(days)],
        'P_COAL': [model.results.var_vals[('P_COAL', t)] for t in range(days)],
        'P_WIND': [model.results.var_vals[('P_WIND', t)] for t in range(days)],
        'P_PV': [model.results.var_vals[('P_PV', t)] for t in range(days)],
        'Q_GAS_BUY': [model.results.var_vals[('Q_GAS_BUY', t)] for t in range(days)],
        'Q_COAL_BUY': [model.results.var_vals[('Q_COAL_BUY', t)] for t in range(days)],
        'Q_EUA_BUY': [model.results.var_vals[('Q_EUA_BUY', t)] for t in range(days)],
        'Q_EUA_SELL': [model.results.var_vals[('Q_EUA_SELL', t)] for t in range(days)],
        
    }
    
    # Calulate storage and EUA balances after first half-year
    demand_met = implemented_decisions['P_GAS'] + implemented_decisions['P_COAL'] + implemented_decisions['P_WIND'] + implemented_decisions['P_PV']
    total_demand_met = sum(demand_met)
     
    coal_storage_end = model.results.var_vals[('Q_COAL_STORAGE', days - 1)]
    gas_storage_end = model.results.var_vals[('Q_GAS_STORAGE', days - 1)]
    eua_balance_end = model.results.var_vals[('Q_EUA_BALANCE', days - 1)]
    
    return total_demand_met, coal_storage_end, gas_storage_end, eua_balance_end, implemented_decisions

def second_stage_optimization(n_scenarios: int, input_data: InputData, days: int):
    total_demand_met, coal_storage_end, gas_storage_end, eua_balance_end, implemented_decisions = realized_result()
        
        
    # Build InputData for the remaining horizon (days 182-364)
    remaining_input_data = InputData(
        variables,
        input_data.gas_prices[182:],
        input_data.coal_prices[182:],
        input_data.eua_prices[182:],
        input_data.rhs_demand - total_demand_met,
        input_data.rhs_storage,
        input_data.rhs_prod,  # Pass the whole dictionary (it's not time-dependent)
        input_data.rhs_wind_prod[182:],
        input_data.rhs_pv_prod[182:],
        input_data.efficiencies,
        input_data.co2_per_kWh,
        input_data.min_prod_ratio,
        {
            'Q_COAL_STORAGE': coal_storage_end,
            'Q_GAS_STORAGE': gas_storage_end
        },
        eua_balance_end
    )
    
    model = StochasticModel(remaining_input_data, n_scenario=n_scenarios, days=days)
    model.run()
    
    return model, implemented_decisions
       




if __name__ == "__main__":


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

    #total_demand, coal_storage_end, gas_storage_end, eua_balance_end, implemented_decisions = realized_result()
    model, implemented_decisions = second_stage_optimization(n_scenarios=50, input_data=input_data, days=366-182)
    model._save_results()
    model.display_results()
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
    model.ex_post_analysis()
    # Plot histogram of ex-post objective values
    plot_histogram(
        model.results.ex_post_obj_vals,
        xlabel="Ex Post Objective Value [EUR]",
        ylabel="Frequency of results across out-of-sample scenarios",
        title="Distribution of Ex Post Objective Values Across Out-of-Sample Scenarios",
        bins=50
    )
    