import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

T = 5
a = 2
b = -1/2
num_time_points = 21
t_vals = np.linspace(0, T, num_time_points)

# Load plot data
with np.load(join(dirname(__file__), 'data.npz')) as data:
    expected_position_analytical = data['expected_position_analytical']
    ideal_data = data['x_obs_ham_ebd']
    ionq_data = data['x_obs_ionq']
    ionq_err = data['x_obs_ionq_err']

#fig = plt.figure(figsize=(100/25.4, 100/25.4), dpi=300)
# figure setup
# plt.rcParams['font.family'] = 'Helvetica'

plt.figure()
plt.plot(t_vals, expected_position_analytical, '-s', color="violet", label="Closed-form solution: " + r"$\langle\hat{x}\rangle_t = \frac{1}{4}(1 - \cos(\sqrt{2} t))$", linewidth=1)
#plt.plot(t_vals, x_obs_ham_ebd, '-o', label="Hamiltonian embedding")
#plt.plot(t_vals, ideal_data, 'ro', label="Numerical simulation (5-level subsystem)", markersize=5)
plt.errorbar(t_vals, ionq_data, ionq_err, fmt='--o', color='blue', ecolor='skyblue', label="Experiment on IonQ (one-hot embedding)", capsize=4)
plt.ylabel(r"Expectation value of position observable $\hat{x}$", fontsize=12)
plt.xlabel(r"$t$ (evolution time)", fontsize=12)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.legend(fontsize=11)
plt.ylim(-0.25, 0.64)
plt.title(r"Observable: $\hat{x}$", fontsize=22, y=1.02)
plt.savefig(join(dirname(__file__), 'real_space_ionq_position.pdf'), dpi=300)
# plt.show()