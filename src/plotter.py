import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

def color_palette():
    ### Color Theme ###
    background_color = "#FAEEDD"

    color_palette = [
    "#B00020",  # Deep Crimson (darker than original red)
    "#D62828",  # Red
    "#E04A2E",  # Strong Red-Orange
    "#F26430",  # Vivid Vermilion
    "#F77F4F",  # Coral
    "#F9A45C",  # Bright Warm Orange
    "#FBBF6B",  # Golden Orange
    "#F5C07A",  # Muted Gold
    "#F2D8A0",  # Light Warm Sand
    "#2A9D8F",  # Teal
    "#66C2A4",  # Light Teal    
    "#A0E7D6",  # Very Light Teal
    "#FBBF6B",  # PV - Golden Orange
    "#5C97D9",  # Wind - Blue
    "#21B582",  # Gas - Dark Green
    "#7D7878",  # Coal - Black
]
    return color_palette, background_color



def plot_histogram(data, xlabel, ylabel, title, bins=30):
    colors, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=bins, color=colors[2], edgecolor='black', alpha=0.7)
    # Add expected cost dashed line
    mean_value = sum(data) / len(data)
    ax.axvline(mean_value, color='black', linestyle='dashed', linewidth=1)
    ax.text(mean_value*1.01, ax.get_ylim()[1]*0.9, f'Expected Cost: {mean_value:.2f} EUR', color='black')
    ax.set_xlabel(xlabel)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.0, 1.07, title, transform=ax.transAxes, fontsize=14, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.03, ylabel, transform=ax.transAxes, fontsize=10, color='black', ha='left')
    plt.tight_layout()
    plt.show()

def plot_gas_coal_prices(time_index, coal_prices, gas_prices):
    colors, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(time_index, coal_prices, label='Coal Prices [EUR/KWh]', color=colors[0])
    ax.step(time_index, gas_prices, label='Gas Prices [EUR/KWh]', color=colors[4])
    ax.set_xlabel('Time')
    ax.text(0.0, 1.07, 'Fuel Prices over Time', transform=ax.transAxes, fontsize=14, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.03, 'Fuel Prices for gas and coal for 2024 [EUR/KWh]', transform=ax.transAxes, fontsize=10, color='black', ha='left')
    ax.text(0.0, -0.1, 'Sources: BusinessInsider and Energinet', transform=ax.transAxes, fontsize=8, color='black', ha='left', alpha=0.7)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_eua_prices(time_index, eua_prices):
    colors, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(time_index, eua_prices, label='EUA Prices [EUR/kgCO2eq]', color=colors[0])
    ax.set_xlabel('Time')
    ax.text(0.0, 1.07, 'EUA Prices over Time', transform=ax.transAxes, fontsize=14, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.03, 'EUA Prices for 2024 [EUR/kgCO2eq]', transform=ax.transAxes, fontsize=10, color='black', ha='left')
    ax.text(0.0, -0.1, 'Source: ICAP', transform=ax.transAxes, fontsize=8, color='black', ha='left', alpha=0.7)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    
def plot_renewables(time_index, wind_prod, pv_prod):
    colors, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(time_index, wind_prod, label='Wind Production [KWh]', color=colors[0])
    ax.step(time_index, pv_prod, label='PV Production [KWh]', color=colors[4])
    ax.set_xlabel('Time')
    ax.text(0.0, 1.07, 'Renewable Energy Production over Time', transform=ax.transAxes, fontsize=14, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.03, 'Wind and PV production for 2024 [KWh]', transform=ax.transAxes, fontsize=10, color='black', ha='left')
    ax.text(0.0, -0.1, 'Source: RenewablesNinja (synthetic data)', transform=ax.transAxes, fontsize=8, color='black', ha='left', alpha=0.7)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    
def plot_gas_vs_coal_production(betavalues, gas_prod, coal_prod):
    colors, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(betavalues, gas_prod, label='Gas Capacity Factor', color=colors[0], marker='o')
    ax.plot(betavalues, coal_prod, label='Coal Capacity Factor', color=colors[4], marker='o')
    ax.set_xlabel('Beta Values')
    ax.text(0.0, 1.07, 'Gas Capacity Factor vs Coal Capacity Factor vs Beta Values', transform=ax.transAxes, fontsize=14, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.03, 'Capacity Factors for different Risk Averse Levels', transform=ax.transAxes, fontsize=10, color='black', ha='left')
    ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,                # number of legend columns
    frameon=False
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    
def plot_storage_dual_vs_beta(betavalues, gas_storage_dual, coal_storage_dual):
    colors, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(betavalues, gas_storage_dual, label='Gas Storage Dual Values', color=colors[0], marker='o')
    ax.plot(betavalues, coal_storage_dual, label='Coal Storage Dual Values', color=colors[4], marker='o')
    ax.set_xlabel('Beta Values', fontsize=12)
    ax.text(0.0, 1.07, 'Value of Gas and Coal Storage vs More Risk-Aversion', transform=ax.transAxes, fontsize=16, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.03, 'Average Change of Objective Value across all scenarios [EUR/100MWh]', transform=ax.transAxes, fontsize=14, color='black', ha='left')
    ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,                # number of legend columns
    frameon=False
    )
    ax.yaxis.get_offset_text().set_x(-0.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    
def plot_energy_mix(tot_p_gas, tot_p_coal, tot_p_wind, tot_p_pv):
    
    colors, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Gas', 'Coal', 'Wind', 'PV']
    sizes = [tot_p_gas, tot_p_coal, tot_p_wind, tot_p_pv]
    colors = [colors[0], colors[2], colors[5], colors[7]]
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.text(0.0, 1.07, 'Energy Mix Distribution', transform=ax.transAxes, fontsize=14, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.03, 'Proportion of Energy Production by Source', transform=ax.transAxes, fontsize=10, color='black', ha='left')
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    
def plot_scenario_infeasibility(n_scenarios, infeasible_count):
    colors, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_scenarios, infeasible_count, marker='o', color=colors[0])
    ax.set_xlabel('Number of Total Scenarios', fontsize=12)
    ax.text(0.0, 1.07, 'Scenario Infeasibility Analysis', transform=ax.transAxes, fontsize=18, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.03, 'Fail Rate, number of infeasible out-of-sample scenarios / out-of-sample scenarios', transform=ax.transAxes, fontsize=14, color='black', ha='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    

def plot_storage_usage(storage_price, storage_level_gas, storage_level_coal):
    colors, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # ---- Smooth interpolation ----
    x_new = np.linspace(np.min(storage_price), np.max(storage_price), 300)
    gas_smooth  = np.interp(x_new, storage_price, storage_level_gas)
    coal_smooth = np.interp(x_new, storage_price, storage_level_coal)

    # Smooth lines
    ax.plot(x_new, gas_smooth,  label='Gas Storage Level [KWh]',  color=colors[0])
    ax.plot(x_new, coal_smooth, label='Coal Storage Level [KWh]', color=colors[4])

    # Original data points (optional but nice)
    ax.scatter(storage_price, storage_level_gas,  color=colors[0], s=20)
    ax.scatter(storage_price, storage_level_coal, color=colors[4], s=20)
    # Labels & title
    ax.set_xlabel('Storage Cost [EUR/kWh]', fontsize=12)
    ax.set_ylabel('Storage Level [KWh]', fontsize=12)

    ax.text(
        0.0, 1.07,
        'Total Stored Amount of Fuel vs. Storage Costs',
        transform=ax.transAxes, fontsize=18, color='black',
        ha='left', fontweight='bold'
    )
    ax.text(
        0.0, 1.02,
        'Total Stored Amount of Fuel over a Year [KWh]',
        transform=ax.transAxes, fontsize=14, color='black',
        ha='left'
    )

    # Legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False
    )

    # Clean up top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def plot_fuel_purchases_over_time(time_index, total_coal_purchases_dict, total_gas_purchases_dict):
    colors, background_color = color_palette()
    background_color = "white"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use colors from color_palette_energy_mix
    multipliers = sorted(total_coal_purchases_dict.keys())
    colors = [colors[i % len(colors)] for i in range(len(multipliers))]
    
    for (eua_multiplier, total_coal_purchases), c in zip(total_coal_purchases_dict.items(), colors):
        ax.plot(time_index, total_coal_purchases, label=f'Coal Purchases (EUA x{eua_multiplier})', linestyle='--', color=c)

    for (eua_multiplier, total_gas_purchases), c in zip(total_gas_purchases_dict.items(), colors):
        ax.plot(time_index, total_gas_purchases, label=f'Gas Purchases (EUA x{eua_multiplier})', linestyle='-', color=c)

    ax.set_xlabel('Time (day)')
    ax.set_ylabel('Energy Purchases [kWh]')
    ax.text(0.0, 1.14, 'Total Fuel Purchases Over Time', transform=ax.transAxes, fontsize=18, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.08, 'Cumulative Coal and Gas Purchases for Different EUA Price Scenarios', transform=ax.transAxes, fontsize=14, color='black', ha='left')
    ax.text(0.0, -0.18, 'Note: Dashed lines represent Coal Purchases, Solid lines represent Gas Purchases', transform=ax.transAxes, fontsize=12, color='black', ha='left', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax.legend()
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()


