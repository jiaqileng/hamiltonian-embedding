import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load plot data
with np.load('data.npz') as data:
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
#ax.set_title('Numerical')
ax.set_xlabel('Evolution time', fontsize=14)
ax.set_ylabel('Layer', fontsize=14)
#plt.show()
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
#ax.set_title('IonQ', fontsize=14)
ax.set_xlabel('Evolution time', fontsize=14)
ax.set_ylabel('Layer', fontsize=14)
#plt.show()
plt.savefig('ionq_heatmap.png', dpi=300)
'''

# figure setup
plt.rcParams['font.family'] = 'Helvetica'
f, axes = plt.subplots(1, 4, figsize=(160/25.4, 37/25.4), dpi=300)
plt.subplots_adjust(left=0.07, right=0.95, top=0.8, bottom=0.2)


for i in range(len(vis_times)):
    t = vis_times[i]
    idx = vis_index[i]
    N_subspace = subspace_sample_num[idx]

    ideal_level_prob = np.zeros(N_levels)
    ionq_level_prob = np.zeros(N_levels)
    for j in range(N_levels):
        ideal_level_prob[j] = np.sum(ideal_data[idx, level_nodes[j]])
        ionq_level_prob[j] = np.sum(ionq_data[idx, level_nodes[j]])


    ax = axes[i]
    ax.bar(level_index - width/2, ionq_level_prob, width, color='royalblue', label='IonQ')
    ax.bar(level_index + width/2, ideal_level_prob, width, color='forestgreen', label='Numerical')
    ax.set_title(f"T = {t}", fontsize=TITLE_FONT, pad=2)
    
    ax.xaxis.set_tick_params(pad=2)
    ax.yaxis.set_tick_params(pad=2)
    ax.set_xticks([1, 2, 3, 4, 5, 6], labels=['1','2','3','4','5','6'], fontsize=TICK_FONT)
    ax.set_xlabel('Levels', fontsize=LABEL_FONT)
        
    
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], labels=['0', '0.25', '0.5', '0.75', '1'], fontsize=TICK_FONT)
    
    if i in [0]:
        ax.set_ylabel('Distribution', fontsize=LABEL_FONT)


    ax.set_xlim([0.5,6.5])
    ax.set_ylim([0,1])
    ax.legend(loc='upper right', fontsize=LEGEND_FONT, frameon=True, facecolor='white')
    
#plt.show()
plt.savefig('level_prob.png', dpi=300)
'''