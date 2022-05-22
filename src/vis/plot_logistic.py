from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def logistic(bs, bq):
    return 1/(1+np.exp(-(bs - bq)))

bs = np.linspace(-5, 5, 50)

bq = -1
y1 = logistic(bs, bq)

bq = 0
y2 = logistic(bs, bq)

bq = 1
y3 = logistic(bs, bq)

plt.plot(bs, y2, label='Question 1 (bq = -1)')
plt.plot(bs, y1, label='Question 2 (bq = 0)')
plt.plot(bs, y3, label='Question 3 (bq = 1)')
plt.xlabel('Student ability (bs)')
plt.ylabel('Probability of correct response')
plt.legend()
plt.show()


bs = np.linspace(-5, 5, 50)
bq = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(bs, bq)
Z = logistic(X, Y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()