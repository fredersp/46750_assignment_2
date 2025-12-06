import matplotlib.pyplot as plt

obj_vals = {
    'Deterministic': 32222731,
    'Stochastic': 35291019,
    'Risk Averse Stochastic (\u03B2 = 0.95)': 35338875,
    '2-Stage Stochastic': 34855924,
    '4-Stage Stochastic': 34652536,
    '12-Stage Stochastic': 34502615,
    
}

def color_palette_objective():
    ### Color Theme ###
    background_color = "#FAEEDD"

    color_palette = [
    "#D62828",  # Red
    "#F77F4F",  # Coral
    "#FBBF6B",  # Golden Orange
    #add 3 green colars in the same tone
    "#2A9D8F",  # Teal
    "#66C2A4",  # Light Teal    
    "#A0E7D6",  # Very Light Teal

]
    return color_palette, background_color



colors, background_color = color_palette_objective()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(list(obj_vals.keys()), list(obj_vals.values()), color=colors[:len(obj_vals)])
ax.set_xlabel('Expected annual cost [€]', fontsize=12)
ax.tick_params(axis='both', labelsize=12)
ax.text(0.0, 1.07, 'Model Comparison', transform=ax.transAxes, fontsize=18, color='black', ha ='left', fontweight='bold')
ax.text(0.0, 1.03, 'Expected annual cost [€] for each model simulation', transform=ax.transAxes, fontsize=14, color='black', ha ='left')        
ax.set_xlim(left=0, right=max(obj_vals.values())*1.16)
for bar in bars:
    width = bar.get_width()
    ax.annotate(f'€{width:,.0f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),  # 3 points horizontal offset
                textcoords="offset points",
                ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.show()


def color_palette_energy_mix():
    ### Color Theme ###
    background_color = "#FAEEDD"

    economist_warm = [
    "#FBBF6B",  # PV - Golden Orange
    "#5C97D9",  # Wind - Blue
    "#21B582",  # Gas - Dark Green
    "#7D7878",  # Coal - Black
]
    return economist_warm, background_color


categories = ['Stochastic', 'Deterministic']
pv = [9.2, 11.5]
wind = [9.1, 11.3]
gas = [42.2, 41.8]
coal = [39.5, 35.4]

    # Create horizontal stacked bar chart
colors_energy, background_color = color_palette_energy_mix()
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(categories, pv, color=colors_energy[0], label='PV')
ax.barh(categories, wind, left=pv, color=colors_energy[1], label='Wind')
ax.barh(categories, gas, left=[pv[i] + wind[i] for i in range(len(pv))], color=colors_energy[2], label='Gas')
ax.barh(categories, coal, left=[pv[i] + wind[i] + gas[i] for i in range(len(pv))], color=colors_energy[3], label='Coal')

    # Add labels to each segment
for i, category in enumerate(categories):
    ax.text(pv[i]/2, i, f'{pv[i]}%', va='center', ha='center', fontsize=12, fontweight='bold')
    ax.text(pv[i] + wind[i]/2, i, f'{wind[i]}%', va='center', ha='center', fontsize=12, fontweight='bold')
    ax.text(pv[i] + wind[i] + gas[i]/2, i, f'{gas[i]}%', va='center', ha='center', fontsize=12, fontweight='bold')
    ax.text(pv[i] + wind[i] + gas[i] + coal[i]/2, i, f'{coal[i]}%', va='center', ha='center', fontsize=12, fontweight='bold', color='white')

ax.set_xlabel('Percentage of total energy production [%]', fontsize=12)
ax.tick_params(axis='both', labelsize=12)
ax.text(0.0, 1.07, 'Energy Mix Comparison', transform=ax.transAxes, fontsize=18, color='black', ha='left', fontweight='bold')
ax.text(0.0, 1.03, 'Percentage of total energy production: Deterministic vs stochastic', transform=ax.transAxes, fontsize=14, color='black', ha='left')        
ax.set_xlim(left=0, right=100)
ax.legend(fontsize=12)
plt.tight_layout()
plt.show()


