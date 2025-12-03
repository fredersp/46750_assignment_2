from stochastic_opt_model import InputData, StochasticModel
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from data_preparation import DataPreparationCSV, DataPreparationJSON
from stochastic_opt_model import StochasticModel, InputData
import random
import numpy as np
from plotter import plot_histogram, plot_time_series

random.seed(42)


def generate_remaining_scenarios(base: InputData, start_idx: int, n_scenarios: int):
    """Generate new scenarios for the remaining horizon [start_idx, end)."""
    end_idx = len(base.gas_prices)
    span = range(start_idx, end_idx)

    def sample_factor(mu=0.0, sigma=0.1, lower=0.8, upper=1.2):
        return max(lower, min(upper, 1 + random.gauss(mu, sigma)))

    scenarios = []
    for _ in range(n_scenarios):
        daily_shocks = [sample_factor() for _ in span]
        wind = [min(base.rhs_wind_prod[t] * daily_shocks[i], 24 * 150_000) for i, t in enumerate(span)]
        pv = [min(base.rhs_pv_prod[t] * daily_shocks[i], 24 * 100_000) for i, t in enumerate(span)]
        scenarios.append(
            {
                "gas_prices_rem": [base.gas_prices[t] * daily_shocks[i] for i, t in enumerate(span)],
                "coal_prices_rem": [base.coal_prices[t] * daily_shocks[i] for i, t in enumerate(span)],
                "eua_prices_rem": [base.eua_prices[t] * daily_shocks[i] for i, t in enumerate(span)],
                "rhs_wind_prod_rem": wind,
                "rhs_pv_prod_rem": pv,
            }
        )
    return scenarios


def simulate_state_halfyear(model: StochasticModel, realized):
    """Advance storage/EUA balances through first half-year using implemented decisions + realized prices.
    realized: dict with keys 'gas_prices', 'coal_prices', 'eua_prices', 'rhs_wind_prod', 'rhs_pv_prod'
             lists of length half_days (same ordering as model.days[:half_days])
    Returns updated storage levels at day half_days-1 and (optionally) EUA balance.
    """
    half_days = len(realized['gas_prices'])
    # start from data.starting_storage_levels
    gas = model.data.starting_storage_levels['Q_GAS_STORAGE']
    coal = model.data.starting_storage_levels['Q_COAL_STORAGE']
    eua_bal = 0.0
    delivered = 0.0

    for t in range(half_days):
        # implemented decisions from the first solve (take stage-0 decisions)
        pg = model.results.var_vals[('P_GAS', t)]
        pc = model.results.var_vals[('P_COAL', t)]
        q_gb = model.results.var_vals[('Q_GAS_BUY', t)]
        q_cb = model.results.var_vals[('Q_COAL_BUY', t)]
        q_eua_b = model.results.var_vals.get(('Q_EUA_BUY', t), 0.0)
        q_eua_s = model.results.var_vals.get(('Q_EUA_SELL', t), 0.0)
        delivered += pg + pc + model.results.var_vals.get(('P_WIND', t), 0.0) + model.results.var_vals.get(('P_PV', t), 0.0)

        gas = q_gb + gas - pg / model.data.efficiencies['eta_GAS']
        coal = q_cb + coal - pc / model.data.efficiencies['eta_COAL']
        eua_bal += q_eua_b - q_eua_s  # adjust if your formulation differs

    demand_remaining = max(model.data.rhs_demand - delivered, 0)

    return {
        'Q_GAS_STORAGE': gas,
        'Q_COAL_STORAGE': coal,
        'Q_EUA_BALANCE': eua_bal
    }, demand_remaining

def build_halfyear_input(base: InputData, updated_storage, demand_remaining, realized_halfyear, n_days_remaining):
    """Create InputData for remaining horizon with updated storage and new scenarios forward."""

    return InputData(
        variables=base.variables,
        gas_prices=realized_halfyear['gas_prices_rem'],   # or newly generated scenarios downstream
        coal_prices=realized_halfyear['coal_prices_rem'],
        eua_prices=realized_halfyear['eua_prices_rem'],
        rhs_demand=demand_remaining,
        rhs_storage=base.rhs_storage,
        rhs_prod=base.rhs_prod,
        rhs_prod_wind=realized_halfyear['rhs_wind_prod_rem'],
        rhs_prod_pv=realized_halfyear['rhs_pv_prod_rem'],
        efficiencies=base.efficiencies,
        co2_per_kWh=base.co2_per_kWh,
        min_prod_ratio=base.min_prod_ratio,
        starting_storage_levels={
            'Q_GAS_STORAGE': updated_storage['Q_GAS_STORAGE'],
            'Q_COAL_STORAGE': updated_storage['Q_COAL_STORAGE']
        },
        starting_eua_balance=updated_storage['Q_EUA_BALANCE']
    )

def run_two_step(base_input: InputData, n_scenarios_full=20, n_scenarios_half=20):
    days_full = len(base_input.gas_prices)
    half_days = days_full // 2

    # 1) Full-year solve
    m_full = StochasticModel(base_input, days=days_full, n_scenario=n_scenarios_full)
    m_full.run()

    # 2) Observe reality for first half (replace with actual realized trajectories)
    realized_first_half = {
        'gas_prices': base_input.gas_prices[:half_days],   # replace with observed
        'coal_prices': base_input.coal_prices[:half_days],
        'eua_prices': base_input.eua_prices[:half_days],
        'rhs_wind_prod': base_input.rhs_wind_prod[:half_days],
        'rhs_pv_prod': base_input.rhs_pv_prod[:half_days],
    }

    # 3) Advance state through first half using implemented decisions + realized data
    updated_storage, demand_remaining = simulate_state_halfyear(m_full, realized_first_half)

    # 4) Build remaining-horizon InputData (use new scenarios forward)
    remaining = {
        'gas_prices_rem': base_input.gas_prices[half_days:],   # replace with fresh scenarios if desired
        'coal_prices_rem': base_input.coal_prices[half_days:],
        'eua_prices_rem': base_input.eua_prices[half_days:],
        'rhs_wind_prod_rem': base_input.rhs_wind_prod[half_days:],
        'rhs_pv_prod_rem': base_input.rhs_pv_prod[half_days:],
    }
    n_days_rem = days_full - half_days
    forward_scenarios = generate_remaining_scenarios(base_input, half_days, n_scenarios_half)

    # 5) Re-solve from mid-year onward with new scenarios (one model per forward scenario)
    half_models = []
    for scen_rem in forward_scenarios:
        half_input = build_halfyear_input(base_input, updated_storage, demand_remaining, scen_rem, n_days_rem)
        m_half = StochasticModel(half_input, days=n_days_rem, n_scenario=1)
        m_half.run()
        half_models.append(m_half)

    return m_full, half_models


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

    starting_eua_balance = 0.0  # Adjust as needed
    
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
    
    base_input = input_data  # build your InputData as you do now
    m_full, half_models = run_two_step(base_input, n_scenarios_full=2, n_scenarios_half=3)
    m_full.display_results()
    if half_models:
        half_models[0].display_results()
