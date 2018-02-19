import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np





def rosenbrock(X):
    A = np.zeros(X.shape[1])
    # A = np.ones(X.shape[1])
    B = 10 * np.ones(X.shape[1])
    return np.square(A - X[0]) + B * np.square((X[1] - np.square(X[0])))


def rosenbrock_grad(X):
    A = np.zeros(X.shape[1])
    # A = np.ones(X.shape[1])
    B = 10 * np.ones(X.shape[1])
    X1 = 2 * X[0] - 2 * A + 4 * B * np.power(X[0], 3) - 4 * B * X[0] * X[1]
    X2 = -2 * B * np.square(X[0]) + 2 * B * X[1]
    return np.array([X1, X2])


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
dim = X.shape
input = np.array([X.flatten(), Y.flatten()])
Z = rosenbrock(input)
Z = Z.reshape(dim)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.9)
# Customize the z axis.
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# Initialise X
X = np.array([[np.random.rand() * 3], [np.random.rand() * 3]])
X = np.array([[-2], [-2]])
# Gradient descent
alpha = 0.00001
GD_plot = np.array([X[0], X[1], [rosenbrock(X)]])

for _ in range(200):
    print(rosenbrock(X))
    X = X - alpha * rosenbrock_grad(X)
    GD_plot = np.append(GD_plot, [X[0], X[1], rosenbrock(X)], axis=1)

ax.scatter(GD_plot[0], GD_plot[1], GD_plot[2], c='b', marker='x')
# PSO initialisation
num_particle = 3
V_own = np.zeros(2).reshape(2,1)
V_best = np.zeros(2).reshape(2,1)
a, b, c = 0.3, 0.3, 0.3
rn = np.random.random
V = np.array([[2], [-2]])
# Particle Swarm Optimisation
PSO_plot = np.array([V[0], V[1], [rosenbrock(V)]])
for _ in range(200):

    print(rosenbrock(V))
    V = a * V + b*rn()*V_own + c*rn()*V_best
    if rosenbrock(V) <= rosenbrock(V_own):
        V_own = V
    if rosenbrock(V) <= rosenbrock(V_best):
        V_best = V
    PSO_plot = np.append(PSO_plot, [V[0], V[1], rosenbrock(V)], axis=1)

ax.scatter(PSO_plot[0], PSO_plot[1], PSO_plot[2], c='r', marker='x')

plt.show()
