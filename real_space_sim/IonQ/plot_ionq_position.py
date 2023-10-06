import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


T = 5
a = 2
b = -1/2
num_time_points = 21
t_vals = np.linspace(0, T, num_time_points)

plt.figure()
x = np.linspace(-2, 3, 100)
potential = 0.5 * a * x **2 + b * x
normalize_c = (1 / np.pi) ** 0.5
init_state = normalize_c * np.exp(- x**2)
plt.plot(x, potential, 'r--', label=r'Potential field: $f(x) = x^2 - \frac{1}{2}x$', linewidth=3)
plt.plot(x, init_state, '-', color='navy', label='Initial wave packet (vacuum state)', linewidth=3)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=12)
bottom, top = plt.ylim()
plt.vlines(x=0, ymin=bottom, ymax=top)
plt.text(0.12,1.2, r"$\langle \hat{x} \rangle = 0$", fontsize=14)
plt.savefig('real_space_potential.png', dpi=300)
# plt.show()


# Load plot data
with np.load('data.npz') as data:
    expected_position_analytical = data['expected_position_analytical']
    ideal_data = data['x_obs_ham_ebd']
    ionq_data = data['x_obs_ionq']
    ionq_err = data['x_obs_ionq_err']


#fig = plt.figure(figsize=(100/25.4, 100/25.4), dpi=300)
# figure setup
# plt.rcParams['font.family'] = 'Helvetica'

plt.figure()
plt.plot(t_vals, expected_position_analytical, '-s', color="violet", label="Closed-form solution: " + r"$<\hat{x}>_t = \frac{1}{4}(1 - \cos(\sqrt{2} t))$", linewidth=1)
#plt.plot(t_vals, x_obs_ham_ebd, '-o', label="Hamiltonian embedding")
#plt.plot(t_vals, ideal_data, 'ro', label="Numerical simulation (5-level subsystem)", markersize=5)
plt.errorbar(t_vals, ionq_data, ionq_err, fmt='--o', color='blue', ecolor='skyblue', label="Experiment on IonQ (one-hot embedding)", capsize=4)
plt.ylabel(r"Expectation value of position observable $\hat{x}$", fontsize=12)
plt.xlabel(r"$t$ (evolution time)", fontsize=12)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.legend(fontsize=11)
plt.savefig('real_space_ionq.png', dpi=300)
# plt.show()