import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join, dirname

# Load plot data
with np.load(join(dirname(__file__), 'data.npz')) as data:
    ideal_data = data['ideal_dist']
    ionq_data = data['ionq_freq_normalized']
    subspace_sample_num = data['num_samples_subspace_ionq']


N_levels = 6
N_nodes = 14
T = 2
num_snapshots = 11
#vis_times = [0, 0.6, 1.2, 1.8]
vis_times = np.linspace(0, T, num_snapshots)
#vis_index = [0, 3, 6, 9]
level_nodes = [[0],[1,2],[3,4,5,6],[7,8,9,10],[11,12],[13]]
level_index = np.arange(1,7)
width = 0.3

TICK_FONT = 5
LEGEND_FONT = 6#
LABEL_FONT = 6
TITLE_FONT = 7

ideal_prob_data = np.zeros((N_levels, num_snapshots))
ionq_prob_data = np.zeros((N_levels, num_snapshots))

for i in range(num_snapshots):
    t = vis_times[i]
    #idx = vis_index[i]
    idx = i
    N_subspace = subspace_sample_num[idx]

    #ideal_level_prob = np.zeros(N_levels)
    #ionq_level_prob = np.zeros(N_levels)
    for j in range(N_levels):
        #ideal_level_prob[j] = np.sum(ideal_data[idx, level_nodes[j]])
        #ionq_level_prob[j] = np.sum(ionq_data[idx, level_nodes[j]])
        ideal_prob_data[j, i] = np.sum(ideal_data[idx, level_nodes[j]])
        ionq_prob_data[j, i] = np.sum(ionq_data[idx, level_nodes[j]])

x_tick_labels = []
for i in range(num_snapshots):
    x_tick_labels.append(f'{vis_times[i]:.1f}')

y_tick_labels = []
for i in range(1, 7):
    y_tick_labels.append(f'{i}')


plt.rcParams['font.family'] = 'Helvetica'

# Ideal heatmap
plt.figure()
ax = plt.gca()
im = ax.imshow(ideal_prob_data,
           cmap='Purples',
           origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
ax.set_xticks(ticks=np.arange(num_snapshots), labels=x_tick_labels)
ax.set_yticks(ticks=np.arange(N_levels), labels=y_tick_labels)
ax.set_xlabel('Evolution time', fontsize=14)
ax.set_ylabel('Layer', fontsize=14)
plt.savefig('numerical_heatmap.png', dpi=300)

# IonQ heatmap
plt.figure()
ax = plt.gca()
im = ax.imshow(ionq_prob_data,
           cmap='Blues',
           origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
ax.set_xticks(ticks=np.arange(num_snapshots), labels=x_tick_labels)
ax.set_yticks(ticks=np.arange(N_levels), labels=y_tick_labels)
ax.set_xlabel('Evolution time', fontsize=14)
ax.set_ylabel('Layer', fontsize=14)
plt.savefig(join(dirname(__file__), 'ionq_heatmap.png'), dpi=300)