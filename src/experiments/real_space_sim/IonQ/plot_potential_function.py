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

plt.figure()
x = np.linspace(-1.5, 2.5, 128)
potential = 0.5 * a * x **2 + b * x
normalize_c = (1 / np.pi) ** 0.5
init_state = normalize_c * np.exp(- x**2)
plt.plot(x, potential, 'r--', label=r'Potential field: $f(x) = x^2 - \frac{1}{2}x$', linewidth=3)
plt.plot(x, init_state, '-', color='navy', label='Initial wave packet (vacuum state)', linewidth=3)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.legend(fontsize=12)
plt.ylim(-0.15, 4)
bottom, top = plt.ylim()
plt.vlines(x=0, ymin=bottom, ymax=2.5)
plt.text(-0.25,2.6, r"$\langle \hat{x} \rangle = 0$", fontsize=14)
plt.savefig(join(dirname(__file__), 'real_space_potential.pdf'), bbox_inches="tight")
# plt.show()
