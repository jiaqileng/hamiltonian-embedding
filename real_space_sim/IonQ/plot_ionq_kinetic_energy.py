import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Load plot data
with np.load('data.npz') as data:
    expected_position_analytical = data['expected_position_analytical']
    ideal_data = 0.5 * data['p2_obs_ham_ebd']
    ionq_data = 0.5 * data['p2_obs_ionq']
    ionq_err = data['kinetic_energy_err']

T = 5
a = 1
b = -1/2
num_time_points = 21
t_vals = np.linspace(0, T, num_time_points)
kinetic_energy_analytical = a/4 - (0.5 * a * expected_position_analytical ** 2 + b * expected_position_analytical)

#fig = plt.figure(figsize=(100/25.4, 100/25.4), dpi=300)
# figure setup
# plt.rcParams['font.family'] = 'Helvetica'

plt.figure()
plt.plot(t_vals, kinetic_energy_analytical, '-s', color="violet", label="Closed-form solution for " + r"$\frac{1}{2}\langle\hat{p}\rangle_t$", linewidth=1)
# plt.plot(t_vals, ideal_data, '-o', color="violet", label="Hamiltonian embedding")
# plt.plot(t_vals, ideal_data, 'ro', label="Numerical simulation (5-level subsystem)", markersize=5)
plt.errorbar(t_vals, ionq_data, ionq_err, fmt='--o', color='blue', ecolor='skyblue', label="Experiment on IonQ (one-hot embedding)", capsize=4)
plt.ylabel(r"Expected kinetic energy $\frac{1}{2}\hat{p}^2$", fontsize=12)
plt.xlabel(r"$t$ (evolution time)", fontsize=12)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.legend(fontsize=11)
plt.savefig('real_space_ionq_kinetic_energy.png', dpi=300)
# plt.show()