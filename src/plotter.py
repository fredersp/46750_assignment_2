import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

def color_palette():
    ### Color Theme ###
    background_color = "#FAEEDD"

    economist_warm = [
    "#B00020",  # Deep Crimson (darker than original red)
    "#D62828",  # Red
    "#E04A2E",  # Strong Red-Orange
    "#F26430",  # Vivid Vermilion
    "#F77F4F",  # Coral
    "#F9A45C",  # Bright Warm Orange
    "#FBBF6B",  # Golden Orange
    "#F5C07A",  # Muted Gold
    "#F2D8A0",  # Light Warm Sand
    #add 3 green colars in the same tone
    "#2A9D8F",  # Teal
    "#66C2A4",  # Light Teal    
    "#A0E7D6",  # Very Light Teal
]
    return economist_warm, background_color



def plot_histogram(data, xlabel, ylabel, title, bins=30):
    economist_warm, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=bins, color=economist_warm[2], edgecolor='black', alpha=0.7)
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
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)
    plt.tight_layout()
    plt.show()

def plot_gas_coal_prices(time_index, coal_prices, gas_prices):
    economist_warm, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(time_index, coal_prices, label='Coal Prices [EUR/KWh]', color=economist_warm[0])
    ax.step(time_index, gas_prices, label='Gas Prices [EUR/KWh]', color=economist_warm[4])
    ax.set_xlabel('Time')
    ax.text(0.0, 1.07, 'Fuel Prices over Time', transform=ax.transAxes, fontsize=14, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.03, 'Fuel Prices for gas and coal for 2024 [EUR/KWh]', transform=ax.transAxes, fontsize=10, color='black', ha='left')
    ax.text(0.0, -0.1, 'Sources: BusinessInsider and Energinet', transform=ax.transAxes, fontsize=8, color='black', ha='left', alpha=0.7)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    plt.tight_layout()
    plt.show()

def plot_eua_prices(time_index, eua_prices):
    economist_warm, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(time_index, eua_prices, label='EUA Prices [EUR/kgCO2eq]', color=economist_warm[0])
    ax.set_xlabel('Time')
    ax.text(0.0, 1.07, 'EUA Prices over Time', transform=ax.transAxes, fontsize=14, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.03, 'EUA Prices for 2024 [EUR/kgCO2eq]', transform=ax.transAxes, fontsize=10, color='black', ha='left')
    ax.text(0.0, -0.1, 'Source: ICAP', transform=ax.transAxes, fontsize=8, color='black', ha='left', alpha=0.7)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    plt.tight_layout()
    plt.show()
    
def plot_renewables(time_index, wind_prod, pv_prod):
    economist_warm, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(time_index, wind_prod, label='Wind Production [KWh]', color=economist_warm[0])
    ax.step(time_index, pv_prod, label='PV Production [KWh]', color=economist_warm[4])
    ax.set_xlabel('Time')
    ax.text(0.0, 1.07, 'Renewable Energy Production over Time', transform=ax.transAxes, fontsize=14, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.03, 'Wind and PV production for 2024 [KWh]', transform=ax.transAxes, fontsize=10, color='black', ha='left')
    ax.text(0.0, -0.1, 'Source: RenewablesNinja (synthetic data)', transform=ax.transAxes, fontsize=8, color='black', ha='left', alpha=0.7)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    plt.tight_layout()
    plt.show()
    
def plot_gas_vs_coal_production(betavalues, gas_prod, coal_prod):
    economist_warm, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(betavalues, gas_prod, label='Gas Capacity Factor', color=economist_warm[0], marker='o')
    ax.plot(betavalues, coal_prod, label='Coal Capacity Factor', color=economist_warm[4], marker='o')
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
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    plt.tight_layout()
    plt.show()
    
def plot_storage_dual_vs_beta(betavalues, gas_storage_dual, coal_storage_dual):
    economist_warm, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(betavalues, gas_storage_dual, label='Gas Storage Dual Values', color=economist_warm[0], marker='o')
    ax.plot(betavalues, coal_storage_dual, label='Coal Storage Dual Values', color=economist_warm[4], marker='o')
    ax.set_xlabel('Beta Values')
    ax.text(0.0, 1.07, 'Value of Gas and Coal Storage vs More Risk-Aversion', transform=ax.transAxes, fontsize=14, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.03, 'Average Dual Values for Max. Storage Capacity Constraints across all scenarios for different Risk Averse Levels [EUR]', transform=ax.transAxes, fontsize=10, color='black', ha='left')
    ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,                # number of legend columns
    frameon=False
    )
    ax.yaxis.get_offset_text().set_x(-0.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    plt.tight_layout()
    plt.show()
    
def plot_energy_mix(tot_p_gas, tot_p_coal, tot_p_wind, tot_p_pv):
    
    economist_warm, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Gas', 'Coal', 'Wind', 'PV']
    sizes = [tot_p_gas, tot_p_coal, tot_p_wind, tot_p_pv]
    colors = [economist_warm[0], economist_warm[2], economist_warm[5], economist_warm[7]]
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.text(0.0, 1.07, 'Energy Mix Distribution', transform=ax.transAxes, fontsize=14, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.03, 'Proportion of Energy Production by Source', transform=ax.transAxes, fontsize=10, color='black', ha='left')
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()