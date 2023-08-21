import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from utils import check_and_make_dir

from scipy.optimize import curve_fit

DATA_DIR = "resource_data"
TASK_DIR = "real_space"

CURR_DIR = DATA_DIR
check_and_make_dir(CURR_DIR)
CURR_DIR = join(CURR_DIR, TASK_DIR)
check_and_make_dir(CURR_DIR)

trotter_method = "second_order"

# Load data
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


from scipy.optimize import curve_fit

# Extrapolation for unary
if trotter_method == "first_order":
    alpha_unary = 2
else:
    alpha_unary = 3/2

# alpha_unary = np.polyfit(np.log(N_vals_unary_bound), np.log(unary_trotter_steps_bound), deg=1)[0]

def func(x, c):
    return c * x ** alpha_unary

c = curve_fit(func, N_vals_unary, unary_trotter_steps)[0][0]
print(f"Real-space simulation")
print(f"Extrapolated curve: f(x) = {c: 0.3f} * x ^ {alpha_unary : 0.3f}")

N_vals_unary_extrapolated = np.arange(N_vals_unary[-1], N_vals_binary[-1]+1)
unary_trotter_steps_extrapolated = func(N_vals_unary_extrapolated, c)

unary_two_qubit_gate_count_per_trotter_step_extrapolated = 2 * (N_vals_unary_extrapolated - 2)
unary_two_qubit_gate_count_per_trotter_step_bound = 2 * (N_vals_unary_bound - 2)

# Extrapolation for one_hot
if trotter_method == "first_order":
    alpha_one_hot = 1
else:
    alpha_one_hot = 1/2
    
# alpha_one_hot = np.polyfit(np.log(N_vals_one_hot_bound), np.log(one_hot_trotter_steps_bound), deg=1)[0]

def func(x, c):
    return c * x ** alpha_one_hot

c = curve_fit(func, N_vals_one_hot, one_hot_trotter_steps)[0][0]
print(f"Extrapolated curve: f(x) = {c: 0.3f} * x ^ {alpha_one_hot : 0.3f}")

N_vals_one_hot_extrapolated = np.arange(N_vals_one_hot[-1], N_vals_binary[-1]+1)
one_hot_trotter_steps_extrapolated = func(N_vals_one_hot_extrapolated, c)
if trotter_method == "first_order" or trotter_method == "randomized_first_order":
    one_hot_two_qubit_gate_count_per_trotter_step_extrapolated = 2 * (N_vals_one_hot_extrapolated - 1)
    one_hot_two_qubit_gate_count_per_trotter_step_bound = 2 * (N_vals_one_hot_bound - 1)
elif trotter_method == "second_order":
    one_hot_two_qubit_gate_count_per_trotter_step_extrapolated = 4 * (N_vals_one_hot_extrapolated - 1)
    one_hot_two_qubit_gate_count_per_trotter_step_bound = 4 * (N_vals_one_hot_bound - 1)

'''
plt.plot(N_vals_binary, binary_trotter_steps * binary_two_qubit_gate_count_per_trotter_step, '-o', label="Standard binary", markersize=3)
# Plot unary
plt.plot(N_vals_unary, unary_trotter_steps * unary_two_qubit_gate_count_per_trotter_step, '-o', color="yellowgreen", label="Unary", markersize=3)
plt.plot(N_vals_unary_extrapolated, unary_trotter_steps_extrapolated * unary_two_qubit_gate_count_per_trotter_step_extrapolated, '--', color="yellowgreen", label="Unary extrapolated", markersize=1)
plt.plot(N_vals_unary_bound, unary_trotter_steps_bound * unary_two_qubit_gate_count_per_trotter_step_bound, '-o', color="green", label="Unary bound", markersize=3)

# Plot one-hot
plt.plot(N_vals_one_hot, one_hot_trotter_steps * one_hot_two_qubit_gate_count_per_trotter_step, '-o', color="orange", label="One-hot", markersize=3)
plt.plot(N_vals_one_hot_extrapolated, one_hot_trotter_steps_extrapolated * one_hot_two_qubit_gate_count_per_trotter_step_extrapolated, '--', color='orange', label="One-hot extrapolated", markersize=1)
plt.plot(N_vals_one_hot_bound, one_hot_trotter_steps_bound * one_hot_two_qubit_gate_count_per_trotter_step_bound, '-o', color="peru", label="One-hot bound", markersize=3)
'''


TICK_FONT = 6
LEGEND_FONT = 8
LABEL_FONT = 9
TITLE_FONT = 8

# figure setup
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.figure(figsize=(100/25.4, 100/25.4), dpi=300)
#plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2)


# std binary
y_data_stdbinary = binary_trotter_steps * binary_two_qubit_gate_count_per_trotter_step
plt.plot(N_vals_binary, y_data_stdbinary, 's', color="red", label="std binary  " + r'$\mathcal{O}\left(n^{2.69}\right)$', markersize=4)
p_std_binary = np.polyfit(np.log(N_vals_binary), np.log(y_data_stdbinary), deg=1)
plt.plot(N_vals_binary, np.exp(p_std_binary[1]) * N_vals_binary **(p_std_binary[0]), 'r-', linewidth=1)
print(f"std binary scaling: {p_std_binary[0]}")

# one-hot theoretical worst bound
y_data_one_hot_bound = one_hot_trotter_steps_bound * one_hot_two_qubit_gate_count_per_trotter_step_bound
p_one_hot_bound = np.polyfit(np.log(N_vals_one_hot_bound), np.log(y_data_one_hot_bound), deg=1)
C_one_hot_bound = y_data_one_hot_bound[0] / (N_vals_one_hot_bound[0] ** p_one_hot_bound[0])
plt.plot(N_vals_one_hot_bound, C_one_hot_bound * N_vals_one_hot_bound **(p_one_hot_bound[0]), '--o', color="skyblue", label="one-hot (theoretical)  " + r'$\mathcal{O}\left(n^{2.86}\right)$', markersize=2, linewidth=0.7)
print(f"one-hot (theoretical) scaling: {p_one_hot_bound[0]}")

# one-hot empirical
y_data_one_hot = one_hot_trotter_steps * one_hot_two_qubit_gate_count_per_trotter_step
plt.plot(N_vals_one_hot, y_data_one_hot, 's', color="blue", label="one-hot (empirical)  " + r'$\mathcal{O}\left(n^{2.05}\right)$', markersize=4)
p_one_hot = np.polyfit(np.log(N_vals_one_hot), np.log(y_data_one_hot), deg=1)
plt.plot(N_vals_binary, np.exp(p_one_hot[1]) * N_vals_binary **(p_one_hot[0]), color='blue', linewidth=1)
print(f"one-hot (empirical) scaling: {p_one_hot[0]}")

plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper left', fontsize=LEGEND_FONT, frameon=True, facecolor='white')
plt.ylabel("Number of 2-qubit gates", fontsize=LABEL_FONT)
plt.xlabel(r"$d$" + " (number of levels in a subsystem)", fontsize=LABEL_FONT)
plt.xticks(fontsize=TICK_FONT)
plt.yticks(fontsize=TICK_FONT)
plt.show()
#plt.savefig('real_space_resource.png', dpi=300)
