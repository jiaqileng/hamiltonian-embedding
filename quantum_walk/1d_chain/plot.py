import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load plot data
with np.load('data.npz') as data:
    ideal_data = data['ideal_dist']
    ionq_data = data['ionq_freq_normalized']
    subspace_sample_num = data['num_samples_subspace_ionq']

N_nodes = 15
T = 4.0
num_snapshots = 17
#vis_times = [0, 0.6, 1.2, 1.8]
vis_times = np.linspace(0, T, num_snapshots)
#vis_index = [0, 3, 6, 9]
width = 0.3

TICK_FONT = 5
LEGEND_FONT = 6
LABEL_FONT = 6
TITLE_FONT = 7

x_tick_labels = []
for i in range(num_snapshots):
    x_tick_labels.append(f'{vis_times[i]:.1f}')

y_tick_labels = []
for i in range(0, N_nodes):
    y_tick_labels.append(f'{i}')


# plt.rcParams['font.family'] = 'Helvetica'

# Ideal heatmap
plt.figure()
ax = plt.gca()
im = ax.imshow(ideal_data.T,
           cmap='plasma',
           origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
ax.set_xticks(ticks=np.arange(num_snapshots)[::2], labels=x_tick_labels[::2])
ax.set_yticks(ticks=np.arange(N_nodes), labels=y_tick_labels)
#ax.set_title('Numerical')
ax.set_xlabel('Evolution time', fontsize=14)
ax.set_ylabel('Vertex', fontsize=14)
#plt.show()
plt.savefig('numerical_heatmap_chain.png', dpi=300)

# IonQ heatmap
plt.figure()
ax = plt.gca()
im = ax.imshow(ionq_data.T,
           cmap='plasma',
           origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
ax.set_xticks(ticks=np.arange(num_snapshots)[::2], labels=x_tick_labels[::2])
ax.set_yticks(ticks=np.arange(N_nodes), labels=y_tick_labels)
#ax.set_title('IonQ', fontsize=14)
ax.set_xlabel('Evolution time', fontsize=14)
ax.set_ylabel('Vertex', fontsize=14)
#plt.show()
plt.savefig('ionq_heatmap_chain.png', dpi=300)
