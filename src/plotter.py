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
    plt.tight_layout()
    plt.show()

def plot_time_series(time, series_dict, title, xlabel, ylabel):
    economist_warm, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (label, series) in enumerate(series_dict.items()):
        ax.step(time, series, label=label, color=economist_warm[i % len(economist_warm)])
    
    ax.set_xlabel(xlabel)
    ax.text(0.0, 1.07, title, transform=ax.transAxes, fontsize=14, color='black', ha='left', fontweight='bold')
    ax.text(0.0, 1.03, ylabel, transform=ax.transAxes, fontsize=10, color='black', ha='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor(background_color)
    ax.legend()
    plt.tight_layout()
    plt.show()