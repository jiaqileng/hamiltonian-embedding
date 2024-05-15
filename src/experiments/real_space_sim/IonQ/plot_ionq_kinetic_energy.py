import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

# Load plot data
with np.load(join(dirname(__file__), 'data.npz')) as data:
    expected_position_analytical = data['expected_position_analytical']
    ideal_data = 0.5 * data['p2_obs_ham_ebd']
    ionq_data = 0.5 * data['p2_obs_ionq']
    ionq_err = data['kinetic_energy_err']

T = 5
a = 2
b = -1/2
num_time_points = 21
t_vals = np.linspace(0, T, num_time_points)
# analytical_exp_x_sq = 5/16 * np.cos(np.sqrt(a) * t_vals)**2 - 0.125 * np.cos(np.sqrt(a) * t_vals) + 5/16
# analytical_exp_x = (b/a) * (np.cos(np.sqrt(a) * t_vals) - 1)
# kinetic_energy_analytical = (1+a)/4 - (0.5 * a * analytical_exp_x_sq + b * analytical_exp_x)

# Plot with higher resolution
plot_every = 5
t_vals_high_res = np.linspace(0, T, (num_time_points - 1) * plot_every + 1)
analytical_exp_x_sq_high_res = 5/16 * np.cos(np.sqrt(a) * t_vals_high_res)**2 - 0.125 * np.cos(np.sqrt(a) * t_vals_high_res) + 5/16
analytical_exp_x_high_res = (b/a) * (np.cos(np.sqrt(a) * t_vals_high_res) - 1)
kinetic_energy_analytical_high_res = (1+a)/4 - (0.5 * a * analytical_exp_x_sq_high_res + b * analytical_exp_x_high_res)

#fig = plt.figure(figsize=(100/25.4, 100/25.4), dpi=300)
# figure setup
# plt.rcParams['font.family'] = 'Helvetica'
plt.figure()
# plt.plot(t_vals, kinetic_energy_analytical, 's-', color="violet", )
plt.plot(t_vals_high_res, kinetic_energy_analytical_high_res, 's-', markevery=range(num_time_points * plot_every)[::plot_every], color="violet", linewidth=1, label="Closed-form solution for " + r"$\frac{1}{2}\langle\hat{p}^2\rangle_t$")
# plt.plot(t_vals, ideal_data, '-o', color="violet", label="Hamiltonian embedding")
# plt.plot(t_vals, ideal_data, 'ro', label="Numerical simulation (5-level subsystem)", markersize=5)
plt.errorbar(t_vals, ionq_data, ionq_err, fmt='--o', color='blue', ecolor='skyblue', label="Experiment on IonQ (one-hot embedding)", capsize=4)
plt.ylabel(r"Expected kinetic energy $\frac{1}{2}\hat{p}^2$", fontsize=12)
plt.xlabel(r"$t$ (evolution time)", fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(0.15, 0.625)
plt.title(r"Observable: $\frac{1}{2}\hat{p}^2$", fontsize=22, y=1.02)
plt.legend(fontsize=11)
plt.savefig(join(dirname(__file__), 'real_space_ionq_kinetic_energy.pdf'), bbox_inches='tight')
# plt.show()