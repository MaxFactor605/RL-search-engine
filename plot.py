import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import seaborn as sns


def read_from_file(filename, data):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        _ = next(reader); _ = next(reader)
        run_data = []
        for i in range(num_runs):
            subdata = []
            for j in range(num_eval_episodes):
                subdata.append(float(next(reader)[0]))
            
            run_data.append(np.mean(subdata))
        data.append(run_data)



designs = {
        'A': {'main_engine_power': 13.0, 'side_engine_power': 0.6},
        'B': {'main_engine_power': 5.0, 'side_engine_power': 2.0},
        'C': {'main_engine_power': 25.0, 'side_engine_power': 0.1},
    }
num_runs = 5
num_eval_episodes = 10

# Plotting parameters
bar_width = 0.15
spacing = 0.2
highlight_color = 'gold'

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting each design's runs
data = []
for i, (design, params)  in enumerate(designs.items()):
    best = False
    
    if os.path.exists("Design_{}_log".format(design)):
        read_from_file("Design_{}_log/eval_env_monitor.log.monitor.csv".format(design), data)
    else:
        best = True
        read_from_file("Design_{}_log_best/eval_env_monitor.log.monitor.csv".format(design), data)
    
    x_positions = np.arange(num_runs) * bar_width + i * (num_runs * bar_width + spacing)
    color = highlight_color if best else sns.color_palette()[i % len(sns.color_palette())]  # Highlight the best design
    ax.bar(x_positions, data[i], width=bar_width, label='Design {} [{}, {}]'.format(design, params["main_engine_power"], params["side_engine_power"]), 
           edgecolor='black', color=color, linewidth=2 if best else 1)


# Setting labels and title
ax.set_ylabel('Performance (reward)', fontsize=14)
ax.set_title('Comparison of Different Designs', fontsize=16, weight='bold')

ax.set_xticks([]) 
ax.spines['bottom'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax.legend(fontsize=12)

#Highlight the best design
for p in ax.patches[:num_runs]:
    p.set_linewidth(3)
    p.set_edgecolor('black')

plt.savefig('design_comparison_chart.png')
plt.tight_layout()
plt.show()