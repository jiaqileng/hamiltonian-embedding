import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from utils import check_and_make_dir

from scipy.optimize import curve_fit


DATA_DIR = "resource_data"
TASK_DIR = "spatial_search_2d"

CURR_DIR = DATA_DIR
check_and_make_dir(CURR_DIR)
CURR_DIR = join(CURR_DIR, TASK_DIR)
check_and_make_dir(CURR_DIR)

dimension = 2

# Load data
trotter_method = "second_order"

resource_estimation_binary = np.load(join(CURR_DIR, f"std_binary_{trotter_method}.npz"))
N_vals_binary = resource_estimation_binary['N_vals_binary']
binary_trotter_steps = resource_estimation_binary['binary_trotter_steps']
binary_two_qubit_gate_count_per_trotter_step = resource_estimation_binary['binary_two_qubit_gate_count_per_trotter_step']

resource_estimation_unary = np.load(join(CURR_DIR, f"unary_{trotter_method}.npz"))
N_vals_unary = resource_estimation_unary["N_vals_unary"]
unary_trotter_steps = resource_estimation_unary['unary_trotter_steps']
unary_two_qubit_gate_count_per_trotter_step = resource_estimation_unary['unary_two_qubit_gate_count_per_trotter_step']

resource_estimation_unary_bound = np.load(join(CURR_DIR, f"unary_{trotter_method}_bound.npz"))
N_vals_unary_bound = resource_estimation_unary_bound["N_vals_unary_bound"]
unary_trotter_steps_bound = resource_estimation_unary_bound['unary_trotter_steps_bound']

resource_estimation_one_hot = np.load(join(CURR_DIR, f"one_hot_{trotter_method}.npz"))
N_vals_one_hot = resource_estimation_one_hot["N_vals_one_hot"]
one_hot_trotter_steps = resource_estimation_one_hot['one_hot_trotter_steps']
one_hot_two_qubit_gate_count_per_trotter_step = resource_estimation_one_hot['one_hot_two_qubit_gate_count_per_trotter_step']

resource_estimation_one_hot_bound = np.load(join(CURR_DIR, f"one_hot_{trotter_method}_bound.npz"))
N_vals_one_hot_bound = resource_estimation_one_hot_bound["N_vals_one_hot_bound"]
one_hot_trotter_steps_bound = resource_estimation_one_hot_bound['one_hot_trotter_steps_bound']


# Extrapolation for unary
if trotter_method == "first_order":
    alpha_unary = 2
else:
    alpha_unary = 3/2

# alpha_unary = np.polyfit(np.log(N_vals_unary_bound), np.log(unary_trotter_steps_bound), deg=1)[0]

def func(x, c):
    return c * x ** alpha_unary

c = curve_fit(func, N_vals_unary, unary_trotter_steps)[0][0]
print(f"Quantum walk on spatial search")
print(f"Extrapolated curve: f(x) = {c: 0.3f} * x ^ {alpha_unary : 0.3f}")

N_vals_unary_extrapolated = np.arange(N_vals_unary[-1], N_vals_binary[-1]+1)
unary_trotter_steps_extrapolated = func(N_vals_unary_extrapolated, c)
if trotter_method == "first_order" or trotter_method == "randomized_first_order":
    unary_two_qubit_gate_count_per_trotter_step_extrapolated = dimension * (N_vals_unary_extrapolated - 2) + 1
    unary_two_qubit_gate_count_per_trotter_step_bound = dimension * (N_vals_unary_bound - 2) + 1
elif trotter_method == "second_order":
    unary_two_qubit_gate_count_per_trotter_step_extrapolated = 2 * dimension * (N_vals_unary_extrapolated - 2) + 1
    unary_two_qubit_gate_count_per_trotter_step_bound = 2 * dimension * (N_vals_unary_bound - 2) + 1
else:
    raise ValueError(f"{trotter_method} not valid")
# Extrapolation for one_hot
if trotter_method == "first_order":
    alpha_one_hot = 1
else:
    alpha_one_hot = 1/2

# alpha_one_hot = np.polyfit(np.log(N_vals_one_hot_bound), np.log(one_hot_trotter_steps_bound), deg=1)[0]

def func(x, c):
    return c * x ** alpha_one_hot

c = curve_fit(func, N_vals_one_hot, one_hot_trotter_steps)[0][0]
print(f"Quantum walk on spatial search")
print(f"Extrapolated curve: f(x) = {c: 0.3f} * x ^ {alpha_one_hot : 0.3f}")

N_vals_one_hot_extrapolated = np.arange(N_vals_one_hot[-1], N_vals_binary[-1]+1)
one_hot_trotter_steps_extrapolated = func(N_vals_one_hot_extrapolated, c)
if trotter_method == "first_order" or trotter_method == "randomized_first_order":
    one_hot_two_qubit_gate_count_per_trotter_step_extrapolated = 2 * dimension * (N_vals_one_hot_extrapolated - 1) + 1
    one_hot_two_qubit_gate_count_per_trotter_step_bound = 2 * dimension * (N_vals_one_hot_bound - 1) + 1
elif trotter_method == "second_order":
    one_hot_two_qubit_gate_count_per_trotter_step_extrapolated = 4 * dimension * (N_vals_one_hot_extrapolated - 1) + 1
    one_hot_two_qubit_gate_count_per_trotter_step_bound = 4 * dimension * (N_vals_one_hot_bound - 1) + 1



TICK_FONT = 5
LEGEND_FONT = 6
LABEL_FONT = 7
TITLE_FONT = 8

# figure setup
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.figure(figsize=(100/25.4, 100/25.4), dpi=300)
#plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2)

# std binary
y_data_stdbinary = binary_trotter_steps * binary_two_qubit_gate_count_per_trotter_step
plt.plot(N_vals_binary, y_data_stdbinary, 's', color="red", label="std binary  " + r'$\mathcal{O}\left(n^{3.60}\right)$', markersize=4)
p_std_binary = np.polyfit(np.log(N_vals_binary), np.log(y_data_stdbinary), deg=1)
plt.plot(N_vals_binary, np.exp(p_std_binary[1]) * N_vals_binary **(p_std_binary[0]), 'r-', linewidth=1)
print(f"std binary scaling: {p_std_binary[0]}")

# unary theoretical worst bound
y_data_unary_bound = unary_trotter_steps_bound * unary_two_qubit_gate_count_per_trotter_step_bound
p_unary_bound = np.polyfit(np.log(N_vals_unary_bound), np.log(y_data_unary_bound), deg=1)
C_unary_bound = y_data_unary_bound[0] / (N_vals_unary_bound[0] ** p_unary_bound[0])
plt.plot(N_vals_unary_bound, C_unary_bound * N_vals_unary_bound **(p_unary_bound[0]), '--o', color="skyblue", label="unary (theoretical)  " + r'$\mathcal{O}\left(n^{3.42}\right)$', markersize=2, linewidth=0.7)
print(f"unary (theoretical) scaling: {p_unary_bound[0]}")

# unary empirical
y_data_unary = unary_trotter_steps * unary_two_qubit_gate_count_per_trotter_step
plt.plot(N_vals_unary, y_data_unary, 's', color="blue", label="unary (empirical)  " + r'$\mathcal{O}\left(n^{3.68}\right)$', markersize=4)
p_unary = np.polyfit(np.log(N_vals_unary), np.log(y_data_unary), deg=1)
plt.plot(N_vals_unary_bound, np.exp(p_unary[1]) * N_vals_unary_bound **(p_unary[0]), color="blue", linewidth=1)
print(f"unary (empirical) scaling: {p_unary[0]}")

# one-hot theoretical worst bound
y_data_one_hot_bound = one_hot_trotter_steps_bound * one_hot_two_qubit_gate_count_per_trotter_step_bound
#plt.plot(N_vals_one_hot_bound, y_data_one_hot_bound, '--o', color="skyblue", label="one-hot (theoretical)  " + r'$\mathcal{O}\left(n^{2.54}\right)$', markersize=2, linewidth=0.7)
p_one_hot_bound = np.polyfit(np.log(N_vals_one_hot_bound), np.log(y_data_one_hot_bound), deg=1)
C_one_hot_bound = y_data_one_hot_bound[0] / (N_vals_one_hot_bound[0] ** p_one_hot_bound[0])
plt.plot(N_vals_one_hot_bound, C_one_hot_bound * N_vals_one_hot_bound **(p_one_hot_bound[0]), '--o', color="violet", label="one-hot (theoretical)  " + r'$\mathcal{O}\left(n^{2.54}\right)$', markersize=2, linewidth=0.7)
print(f"one-hot (theoretical) scaling: {p_one_hot_bound[0]}")

# one-hot empirical
y_data_one_hot = one_hot_trotter_steps * one_hot_two_qubit_gate_count_per_trotter_step
plt.plot(N_vals_one_hot, y_data_one_hot, 's', color="darkviolet", label="one-hot (empirical)  " + r'$\mathcal{O}\left(n^{1.64}\right)$', markersize=4)
p_one_hot = np.polyfit(np.log(N_vals_one_hot), np.log(y_data_one_hot), deg=1)
plt.plot(N_vals_binary, np.exp(p_one_hot[1]) * N_vals_binary **(p_one_hot[0]), color='darkviolet', linewidth=1)
print(f"one-hot (empirical) scaling: {p_one_hot[0]}")

plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper left', fontsize=LEGEND_FONT, frameon=True, facecolor='white')
plt.ylabel("Number of 2-qubit gates", fontsize=LABEL_FONT)
plt.xlabel(r"$N$" + " (number of nodes per edge)", fontsize=LABEL_FONT)
x_tick_labels = []
for i in range(3, 16):
    x_tick_labels.append(f'{i}')
#print(x_tick_labels)
plt.xticks(ticks=np.arange(3, 16), 
           labels=x_tick_labels,
           rotation='horizontal', 
           fontsize=TICK_FONT)
plt.yticks(fontsize=TICK_FONT)
plt.show()
#plt.savefig('spatial_search_2d_resource.png', dpi=300)