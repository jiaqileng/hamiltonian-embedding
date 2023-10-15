import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load plot data
with np.load('data.npz') as data:
    ideal_data = data['ideal_dist']
    ionq_data = data['ionq_freq_normalized']
    subspace_sample_num = data['num_samples_subspace_ionq']

N_nodes = 15
N_levels = int(1 + np.log2(N_nodes))
T = 3
num_snapshots = 16
#vis_times = [0, 0.6, 1.2, 1.8]
vis_times = np.linspace(0, T, num_snapshots)
#vis_index = [0, 3, 6, 9]
width = 0.3

TICK_FONT = 7
LEGEND_FONT = 6
LABEL_FONT = 8
TITLE_FONT = 12

ideal_prob_data = np.zeros((num_snapshots, N_levels))
ionq_prob_data = np.zeros((num_snapshots, N_levels))

j = 0
for i in range(N_levels):
    while j < 2 ** (i+1) - 1:
        ideal_prob_data[:,i] += ideal_data[:,j]
        ionq_prob_data[:,i] += ionq_data[:,j]
        j += 1

x_tick_labels = []
for i in range(num_snapshots):
    x_tick_labels.append(f'{vis_times[i]:.1f}')

y_tick_labels = []
for i in range(1, N_levels+1):
    y_tick_labels.append(f'{i}')


fig, axs = plt.subplots(2, 1, constrained_layout=True, facecolor="white", figsize=(7.5,4.5))

# Ideal heatmap
im = axs[0].imshow(ideal_prob_data.T,
           cmap='Reds',
           origin='lower')
divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
axs[0].set_xticks(ticks=np.arange(num_snapshots), labels=x_tick_labels, fontsize=TICK_FONT)
axs[0].set_yticks(ticks=np.arange(N_levels), labels=y_tick_labels, fontsize=TICK_FONT)
axs[0].set_title('Numerical', fontsize=TITLE_FONT)
axs[0].set_xlabel('Evolution time', fontsize=LABEL_FONT)
axs[0].set_ylabel('Level', fontsize=LABEL_FONT)

# IonQ heatmap
im = axs[1].imshow(ionq_prob_data.T,
           cmap='Reds',
           origin='lower')
divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
axs[1].set_xticks(ticks=np.arange(num_snapshots), labels=x_tick_labels, fontsize=TICK_FONT)
axs[1].set_yticks(ticks=np.arange(N_levels), labels=y_tick_labels, fontsize=TICK_FONT)
axs[1].set_title('IonQ', fontsize=TITLE_FONT)
axs[1].set_xlabel('Evolution time', fontsize=LABEL_FONT)
axs[1].set_ylabel('Level', fontsize=LABEL_FONT)
fig.suptitle("Quantum walk on binary tree", fontsize=14)
# plt.show()
plt.savefig('heatmap_binary_tree.png', dpi=300)