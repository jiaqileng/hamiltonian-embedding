import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import dirname, join
from utils import *

n = 4
h = np.zeros(n)
h[0] = 1
h[-1] = -1
J = np.zeros((n,n))
for i in range(n-1):
    J[i,i + 1] = -1
lamb = 0.1
H = (sum_x(n) + lamb * (sum_h_z(n, h) + sum_J_zz(n, J))).toarray()
codewords = get_codewords_1d(n, encoding="unary", periodic=False)

P = np.zeros((2 ** n, 2 ** n))
for i in range(len(codewords)):
    P[i, codewords[i]] = 1
j = 0
for i in np.arange(len(codewords), 2 ** n):
    while j in codewords:
        j += 1
    P[i, j] = 1
    j += 1

sns.set_theme(style="white")

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(20, 200, s=100, as_cmap=True)
mask = (P @ H @ P.T) == 0
# plt.colorbar().remove()
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap((P @ H @ P.T) - (0.25) * np.eye(2 ** n), mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.savefig(join(dirname( __file__ ), "matrix.png"), transparent=True)
# plt.show()
