import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join, dirname

# Load plot data
with np.load(join(dirname(__file__), 'data.npz')) as data:
    ideal_data = data['ideal_dist']
    ionq_data = data['ionq_freq_normalized']
    subspace_sample_num = data['num_samples_subspace_ionq']

N_nodes = 15
T = 4.0
num_snapshots = 17
vis_times = np.linspace(0, T, num_snapshots)
width = 0.3

TICK_FONT = 5
LEGEND_FONT = 6
LABEL_FONT = 10
TITLE_FONT = 12

x_tick_labels = []
for i in range(num_snapshots):
    x_tick_labels.append(f'{vis_times[i]:.1f}')

y_tick_labels = []
for i in range(0, N_nodes):
    y_tick_labels.append(f'{i+1}')


# plt.rcParams['font.family'] = 'Helvetica'
fig, axs = plt.subplots(2, 1, constrained_layout=True, facecolor="white", figsize=(5.5,9))

# Ideal heatmap
im = axs[0].imshow(ideal_data.T,
           cmap='Reds',
           origin='lower')
divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
axs[0].set_xticks(ticks=np.arange(num_snapshots)[::2], labels=x_tick_labels[::2])
axs[0].set_yticks(ticks=np.arange(N_nodes), labels=y_tick_labels)
axs[0].set_title('Numerical', fontsize=TITLE_FONT)
axs[0].set_xlabel('Evolution time', fontsize=LABEL_FONT)
axs[0].set_ylabel('Vertex', fontsize=LABEL_FONT)

# IonQ heatmap
im = axs[1].imshow(ionq_data.T,
           cmap='Reds',
           origin='lower')
divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
axs[1].set_xticks(ticks=np.arange(num_snapshots)[::2], labels=x_tick_labels[::2])
axs[1].set_yticks(ticks=np.arange(N_nodes), labels=y_tick_labels)
axs[1].set_title('IonQ', fontsize=TITLE_FONT)
axs[1].set_xlabel('Evolution time', fontsize=LABEL_FONT)
axs[1].set_ylabel('Vertex', fontsize=LABEL_FONT)
fig.suptitle("Quantum walk on 1D cycle", fontsize=14)
plt.savefig(join(dirname(__file__), 'heatmap_cycle.png'), dpi=300)
