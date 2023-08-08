import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from utils import check_and_make_dir

from scipy.optimize import curve_fit


DATA_DIR = "resource_data"
TASK_DIR = "glued_tree"

CURR_DIR = DATA_DIR
check_and_make_dir(CURR_DIR)
CURR_DIR = join(CURR_DIR, TASK_DIR)
check_and_make_dir(CURR_DIR)

# Load data
trotter_method = "second_order"

resource_estimation_binary = np.load(join(CURR_DIR, f"std_binary_{trotter_method}.npz"))
N_vals_binary = resource_estimation_binary['N_vals_binary']
binary_trotter_steps = resource_estimation_binary['binary_trotter_steps']
binary_two_qubit_gate_count_per_trotter_step = resource_estimation_binary['binary_two_qubit_gate_count_per_trotter_step']

resource_estimation_one_hot = np.load(join(CURR_DIR, f"one_hot_{trotter_method}.npz"))
N_vals_one_hot = resource_estimation_one_hot["N_vals_one_hot"]
one_hot_trotter_steps = resource_estimation_one_hot['one_hot_trotter_steps']
one_hot_two_qubit_gate_count_per_trotter_step = resource_estimation_one_hot['one_hot_two_qubit_gate_count_per_trotter_step']

resource_estimation_one_hot_bound = np.load(join(CURR_DIR, f"one_hot_{trotter_method}_bound.npz"))
N_vals_one_hot_bound = resource_estimation_one_hot_bound["N_vals_one_hot_bound"]
one_hot_trotter_steps_bound = resource_estimation_one_hot_bound['one_hot_trotter_steps_bound']


# Extrapolation for one_hot
if trotter_method == "first_order":
    alpha_one_hot = 1
else:
    alpha_one_hot = 1/2

def func(x, c):
    return c * x ** alpha_one_hot

# alpha_one_hot = np.polyfit(np.log(N_vals_one_hot_bound), np.log(one_hot_trotter_steps_bound), deg=1)[0]

c = curve_fit(func, N_vals_one_hot, one_hot_trotter_steps)[0][0]
print(f"Quantum walk on glued tree")
print(f"Extrapolated curve: f(x) = {c: 0.3f} * x ^ {alpha_one_hot : 0.3f}")

N_vals_one_hot_extrapolated = np.arange(N_vals_one_hot[-1], N_vals_binary[-1]+1)
one_hot_trotter_steps_extrapolated = func(N_vals_one_hot_extrapolated, c)

if trotter_method == "first_order" or trotter_method == "randomized_first_order":
    one_hot_two_qubit_gate_count_per_trotter_step_extrapolated = 2 * (3 * N_vals_one_hot_extrapolated / 2 - 1)
    one_hot_two_qubit_gate_count_per_trotter_step_bound = 2 * (3 * N_vals_one_hot_bound / 2 - 1)
elif trotter_method == "second_order":
    one_hot_two_qubit_gate_count_per_trotter_step_extrapolated = 4 * (3 * N_vals_one_hot_extrapolated / 2 - 1)
    one_hot_two_qubit_gate_count_per_trotter_step_bound = 4 * (3 * N_vals_one_hot_bound / 2 - 1)


TICK_FONT = 5
LEGEND_FONT = 7
LABEL_FONT = 7
TITLE_FONT = 8

# figure setup
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.figure(figsize=(80/25.4, 80/25.4), dpi=300)
#plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2)

## Standard binary
y_data_stdbinary = binary_trotter_steps * binary_two_qubit_gate_count_per_trotter_step
plt.plot(N_vals_binary, y_data_stdbinary, 's', color="red", label="std binary  " + r'$\mathcal{O}\left(n^{2.60}\right)$', markersize=4)
p_std_binary = np.polyfit(np.log(N_vals_binary), np.log(y_data_stdbinary), deg=1)
plt.plot(N_vals_binary, np.exp(p_std_binary[1]) * N_vals_binary **(p_std_binary[0]), 'r-', linewidth=1)
print(f"std binary scaling: {p_std_binary[0]}")

# One-hot theoretical worst bound
#y_data_extrap = one_hot_trotter_steps_extrapolated * one_hot_two_qubit_gate_count_per_trotter_step_extrapolated
#plt.plot(N_vals_one_hot_extrapolated, y_data_extrap, '--', color='orange', label="One-hot extrapolated", markersize=3)
y_data_bound = one_hot_trotter_steps_bound * one_hot_two_qubit_gate_count_per_trotter_step_bound
plt.plot(N_vals_one_hot_bound, y_data_bound, '--o', color="skyblue", label="one-hot (theoretical)  " + r'$\mathcal{O}\left(n^{1.59}\right)$', markersize=2, linewidth=0.7)
p_bound = np.polyfit(np.log(N_vals_one_hot_bound), np.log(y_data_bound), deg=1)
print(f"one-hot (theoretical) scaling: {p_bound[0]}")

## One-hot empirical
y_data_onehot = one_hot_trotter_steps * one_hot_two_qubit_gate_count_per_trotter_step
plt.plot(N_vals_one_hot, y_data_onehot, 's', color="blue", label="one-hot (empirical)  " + r'$\mathcal{O}\left(n^{1.48}\right)$', markersize=4)
p_onehot = np.polyfit(np.log(N_vals_one_hot), np.log(y_data_onehot), deg=1)
plt.plot(N_vals_binary, np.exp(p_onehot[1]) * N_vals_binary **(p_onehot[0]), 'b-', linewidth=1)
print(f"one-hot (empirical) scaling: {p_onehot[0]}")


plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper left', fontsize=LEGEND_FONT, frameon=True, facecolor='white')
plt.ylabel("Two-qubit Gate Count", fontsize=LABEL_FONT)
plt.xlabel("Number of Vertices on Glued Tree " + r"$(N)$", fontsize=LABEL_FONT)
plt.xticks(fontsize=TICK_FONT)
plt.yticks(fontsize=TICK_FONT)
#plt.grid()
plt.show()
#plt.savefig('glued_tree_resource.png', dpi=300)