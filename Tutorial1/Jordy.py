import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#Parameters
num_iter = 1000
step_size = 0.001
num_particles = 10
#random number range [0] for multiplication and [1] for substraction
rn = np.random.random
rn_range = [4,-2]

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

def rastrigin(X):
    return 10*X.shape[0] + np.square(X) - 10*np.cos(2*np.pi*X)

def rastrigin_grad(X):
    pass

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0+rn_range[1], rn_range[0]+rn_range[1], 0.1)
Y = np.arange(0+rn_range[1], rn_range[0]+rn_range[1], 0.1)
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
# X = np.array([[rn()*rn_range[0]+rn_range[1]], 
#     [np.random.rand()*rn_range[0]+rn_range[1]]])
X = np.array([[rn_range[1]], [rn_range[1]]])
# Gradient descent

GD_plot = np.array([X[0], X[1], [rosenbrock(X)]])

for _ in range(num_iter):

    X = X - step_size * rosenbrock_grad(X)
    GD_plot = np.append(GD_plot, [X[0], X[1], rosenbrock(X)], axis=1)

ax.scatter(GD_plot[0], GD_plot[1], GD_plot[2], c='b', marker='x')

# PSO initialisation particle_best (Sp_best), group_best(Sg_best), 
#   particle(Sp), particle_velocity(Vp)
Sp = np.zeros((num_particles,2))
Sp_best = np.zeros((num_particles,2))
Sg_best = np.zeros(2).reshape(1,2)
Vp = np.zeros((num_particles,2))

a, b, c = 0.3, 0.3, 0.3

PSO_plot = []
for i in range(num_particles):
    Sp[i] = np.array([rn()*rn_range[0]+rn_range[1], 
        rn()*rn_range[0]+rn_range[1]])
    # Sp[i] = np.array([rn_range[1], rn_range[1]])
    if PSO_plot == []:
        PSO_plot = np.array([[Sp[i][0]], [Sp[i][1]], [rosenbrock(Sp[i].reshape(2,1))]])
    else:
        PSO_plot = np.hstack((PSO_plot, np.array([[Sp[i][0]], [Sp[i][1]], [rosenbrock(Sp[i].reshape(2,1))]])))
    
# Particle Swarm Optimisation
for _ in range(num_iter):
    
    for i in range(num_particles):
        
        Vp[i] = a * Vp[i] + b*rn()*Sp_best[i] + c*rn()*Sg_best
        Sp[i] = Sp[i] * Vp[i]
        if rosenbrock(Sp[i].reshape(2,1)) <= rosenbrock(Sp_best[i].reshape(2,1)):
            Sp_best[i] = Sp[i]

        if rosenbrock(Sp[i].reshape(2,1)) <= rosenbrock(Sg_best.reshape(2,1)):
            Sg_best = Sp[i]
                
        PSO_plot = np.hstack((PSO_plot,[[Sp[i][0]],[Sp[i][1]],[rosenbrock(Sp[i].reshape(2,1))]]))

fmt = '{:<12}{:<25}{}'
print(fmt.format("Iteration", "Gradient descent ", "Particle Swarm Optimisation"))
[print(fmt.format(i, gd, pso)) for i, (gd, pso) in enumerate(zip(GD_plot[2], PSO_plot[2]))]
ax.scatter(PSO_plot[0], PSO_plot[1], PSO_plot[2], c='r', marker='x')
plt.show()