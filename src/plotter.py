import seaborn as sns
import matplotlib.pyplot as plt

def color_palette():
    ### Color Theme ###
    background_color = "#F5F0E6"

    economist_warm = [
        "#D62828",  # Red
        "#DF3B2E",  # Mid Red-Orange
        "#E85D3F",  # Vermilion
        "#F77F4F",  # Coral
        "#FBB182",  # Pale Orange
        "#F5C07A",  # Muted Gold
        "#F5F0E6",  # Background Cream
        ]
    return economist_warm, background_color

def plot_histogram(data, xlabel, ylabel, title, bins=30):
    economist_warm, background_color = color_palette()
    
    fig, ax = plt.subplots(figsize=(10, 6))
   
    ax.hist(data, bins=bins, color=economist_warm[2], edgecolor='black', alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.text(0.0, 1.07, title, transform=ax.transAxes, fontsize=14, color='black', ha ='left')
    ax.text(0.0, 1.02, ylabel, transform=ax.transAxes, fontsize=10, color='black', ha ='left')
    ax.set_facecolor(background_color)
    plt.tight_layout()
    plt.show()

