import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from utils import check_and_make_dir

# numerically compute unitary fidelity or just set lambda by the theoretical scaling
compute_fidelity_for_lamb = False

DATA_DIR = "resource_data"
TASK_DIR = "spatial_search_3d"

CURR_DIR = DATA_DIR
check_and_make_dir(CURR_DIR)
CURR_DIR = join(CURR_DIR, TASK_DIR)
check_and_make_dir(CURR_DIR)

trotter_method = "second_order"

resource_estimation_binary = np.load(join(CURR_DIR, f"std_binary_{trotter_method}_remove_edges.npz"))
N_vals_binary = resource_estimation_binary['N_vals_binary']
binary_trotter_steps = resource_estimation_binary['binary_trotter_steps']
binary_one_qubit_gate_count_per_trotter_step = resource_estimation_binary['binary_one_qubit_gate_count_per_trotter_step']
binary_two_qubit_gate_count_per_trotter_step = resource_estimation_binary['binary_two_qubit_gate_count_per_trotter_step']
y_data_binary = binary_trotter_steps * (binary_one_qubit_gate_count_per_trotter_step + binary_two_qubit_gate_count_per_trotter_step)

if compute_fidelity_for_lamb:
    resource_estimation_unary = np.load(join(CURR_DIR, f"unary_{trotter_method}.npz"))
else:
    resource_estimation_unary = np.load(join(CURR_DIR, f"unary_{trotter_method}_set_lamb.npz"))
N_vals_unary = resource_estimation_unary["N_vals_unary"]
unary_trotter_steps = resource_estimation_unary['unary_trotter_steps']
unary_one_qubit_gate_count_per_trotter_step = resource_estimation_unary['unary_one_qubit_gate_count_per_trotter_step']
unary_two_qubit_gate_count_per_trotter_step = resource_estimation_unary['unary_two_qubit_gate_count_per_trotter_step']
y_data_unary = unary_trotter_steps * (unary_one_qubit_gate_count_per_trotter_step + unary_two_qubit_gate_count_per_trotter_step)

if compute_fidelity_for_lamb:
    resource_estimation_unary_bound = np.load(join(CURR_DIR, f"unary_{trotter_method}_bound.npz"))
else:
    resource_estimation_unary_bound = np.load(join(CURR_DIR, f"unary_{trotter_method}_bound_set_lamb.npz"))
N_vals_unary_bound = resource_estimation_unary_bound["N_vals_unary_bound"]
unary_trotter_steps_bound = resource_estimation_unary_bound['unary_trotter_steps_bound']
unary_one_qubit_gate_count_per_trotter_step_bound = resource_estimation_unary_bound['unary_one_qubit_gate_count_per_trotter_step_bound']
unary_two_qubit_gate_count_per_trotter_step_bound = resource_estimation_unary_bound['unary_two_qubit_gate_count_per_trotter_step_bound']
y_data_unary_bound = unary_trotter_steps_bound * (unary_one_qubit_gate_count_per_trotter_step_bound + unary_two_qubit_gate_count_per_trotter_step_bound)

resource_estimation_one_hot = np.load(join(CURR_DIR, f"one_hot_{trotter_method}.npz"))
N_vals_one_hot = resource_estimation_one_hot["N_vals_one_hot"]
one_hot_trotter_steps = resource_estimation_one_hot['one_hot_trotter_steps']
one_hot_one_qubit_gate_count_per_trotter_step = resource_estimation_one_hot['one_hot_one_qubit_gate_count_per_trotter_step']
one_hot_two_qubit_gate_count_per_trotter_step = resource_estimation_one_hot['one_hot_two_qubit_gate_count_per_trotter_step']
y_data_one_hot = one_hot_trotter_steps * (one_hot_one_qubit_gate_count_per_trotter_step + one_hot_two_qubit_gate_count_per_trotter_step)

resource_estimation_one_hot_bound = np.load(join(CURR_DIR, f"one_hot_{trotter_method}_bound.npz"))
N_vals_one_hot_bound = resource_estimation_one_hot_bound["N_vals_one_hot_bound"]
one_hot_trotter_steps_bound = resource_estimation_one_hot_bound['one_hot_trotter_steps_bound']
one_hot_one_qubit_gate_count_per_trotter_step_bound = resource_estimation_one_hot_bound['one_hot_one_qubit_gate_count_per_trotter_step_bound']
one_hot_two_qubit_gate_count_per_trotter_step_bound = resource_estimation_one_hot_bound['one_hot_two_qubit_gate_count_per_trotter_step_bound']
y_data_one_hot_bound = one_hot_trotter_steps_bound * (one_hot_one_qubit_gate_count_per_trotter_step_bound + one_hot_two_qubit_gate_count_per_trotter_step_bound)


TICK_FONT = 5
LEGEND_FONT = 6
LABEL_FONT = 7
TITLE_FONT = 8

# figure setup
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.figure(figsize=(100/25.4, 100/25.4), dpi=300)
#plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2)

p_std_binary = np.polyfit(np.log(N_vals_binary), np.log(y_data_binary), deg=1)
y_data_binary_fit = np.exp(p_std_binary[1]) * N_vals_binary **(p_std_binary[0])
plt.plot(N_vals_binary, y_data_binary, 's', color="red", label=fr"std binary (empirical) $\mathcal{{O}}\left(n^{{{p_std_binary[0]:0.2f}}}\right)$", markersize=4)
plt.plot(N_vals_binary, np.exp(p_std_binary[1]) * N_vals_binary **(p_std_binary[0]), 'r-', linewidth=1)
print(f"std binary scaling: {p_std_binary[0]}")

# Unary theoretical worst bound
p_unary_bound = np.polyfit(np.log(N_vals_unary_bound), np.log(y_data_unary_bound), deg=1)
plt.plot(N_vals_unary_bound, y_data_one_hot_bound, 'o', color="skyblue", label=fr"unary (theoretical) $\mathcal{{O}}\left(n^{{{p_unary_bound[0]:0.2f}}}\right)$", markersize=2)
plt.plot(N_vals_unary_bound, np.exp(p_unary_bound[1]) * N_vals_binary **(p_unary_bound[0]), '--', color="skyblue", linewidth=0.7)
print(f"unary (theoretical) scaling: {p_unary_bound[0]}")

# Unary empirical
p_unary = np.polyfit(np.log(N_vals_unary), np.log(y_data_unary), deg=1)
plt.plot(N_vals_unary, y_data_unary, 's', color="blue", label=fr"unary (empirical) $\mathcal{{O}}\left(n^{{{p_unary[0]:0.2f}}}\right)$", markersize=4)
plt.plot(N_vals_binary, np.exp(p_unary[1]) * N_vals_binary **(p_unary[0]), '-', color="blue", linewidth=1)
print(f"unary (empirical) scaling: {p_unary[0]}")

# One-hot theoretical worst bound
#y_data_extrap = one_hot_trotter_steps_extrapolated * one_hot_two_qubit_gate_count_per_trotter_step_extrapolated
#plt.plot(N_vals_one_hot_extrapolated, y_data_extrap, '--', color='orange', label="One-hot extrapolated", markersize=3)
p_one_hot_bound = np.polyfit(np.log(N_vals_one_hot_bound), np.log(y_data_one_hot_bound), deg=1)
plt.plot(N_vals_one_hot_bound, y_data_one_hot_bound, 'o', color="violet", label=fr"one-hot (theoretical) $\mathcal{{O}}\left(n^{{{p_one_hot_bound[0]:0.2f}}}\right)$", markersize=2)
plt.plot(N_vals_one_hot_bound, np.exp(p_one_hot_bound[1]) * N_vals_binary **(p_one_hot_bound[0]), '--', color="violet", linewidth=0.7)
print(f"one-hot (theoretical) scaling: {p_one_hot_bound[0]}")

# One-hot empirical
p_one_hot = np.polyfit(np.log(N_vals_one_hot), np.log(y_data_one_hot), deg=1)
plt.plot(N_vals_one_hot, y_data_one_hot, 's', color="darkviolet", label=fr"one-hot (empirical) $\mathcal{{O}}\left(n^{{{p_one_hot[0]:0.2f}}}\right)$", markersize=4)
plt.plot(N_vals_binary, np.exp(p_one_hot[1]) * N_vals_binary **(p_one_hot[0]), '-', color="darkviolet", linewidth=1)
print(f"one-hot (empirical) scaling: {p_one_hot[0]}")

plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper left', fontsize=LEGEND_FONT, frameon=True, facecolor='white')
plt.ylabel("Total gate count", fontsize=LABEL_FONT)
plt.xlabel(r"$N$" + " (number of nodes per edge)", fontsize=LABEL_FONT)
x_tick_labels = []
for i in N_vals_binary:
    x_tick_labels.append(f'{i}')
#print(x_tick_labels)
plt.xticks(ticks=N_vals_binary, 
           labels=x_tick_labels,
           rotation='horizontal', 
           fontsize=TICK_FONT)
plt.yticks(fontsize=TICK_FONT)
# plt.show()
plt.savefig(join("resource_estimation_plots", 'spatial_search_3d_resource.png'), dpi=300)