import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def rosenbrock(X):
    A = np.zeros(X.shape[1])
    B = 10 * np.ones(X.shape[1])
    return np.square(A - X[0]) + B * np.square((X[1] - np.square(X[0])))


def rosenbrock_grad(X):
    A = np.zeros(X.shape[1])
    B = 10 * np.ones(X.shape[1])
    X1 = 2 * X[0] - 2 * A + 4 * B * np.power(X[0], 3) - 4 * B * X[0] * X[1]
    X2 = -2 * B * np.square(X[0]) + 2 * B * X[1]
    return np.array([X1, X2])

def rastrigin(X):
    A = np.ones(X.shape[1]) * 10
    
    return A*X.shape[0] + np.sum(np.square(X) - 10*np.cos(2*np.pi*X),axis=0)

def rastrigin_grad(X):
    X1 = 2*X[0] + 10*np.sin(2*np.pi*X[0])*2*np.pi
    X2 = 2*X[1] + 10*np.sin(2*np.pi*X[1])*2*np.pi
    return np.array([X1, X2])

def plotSurface(rn_range, benchmark_function):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data points for mesh grid
    X = np.arange(0+rn_range[1], rn_range[0]+rn_range[1], 0.1)
    Y = np.arange(0+rn_range[1], rn_range[0]+rn_range[1], 0.1)
    X, Y = np.meshgrid(X, Y)
    dim = X.shape
    
    inputVar = np.array([X.flatten(), Y.flatten()])
    Z = benchmark_function(inputVar)
    
    Z = Z.reshape(dim)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.9)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # Customize the z axis.
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    return ax


def gradientDescent(num_iter, rn_range, benchmark_function, benchmark_function_grad):
    # Initialise X
    X = np.array([[rn()*rn_range[0]+rn_range[1]], 
        [np.random.rand()*rn_range[0]+rn_range[1]]])
    # X = np.array([[rn_range[1]], [rn_range[1]]])
    # Gradient descent
    output = np.array([X[0], X[1], [benchmark_function(X)]])
    for _ in range(num_iter):
        X = X - step_size * benchmark_function_grad(X)
        output = np.append(output, [X[0], X[1], benchmark_function(X)], axis=1)
    return output

def particleSwarmOptimisation(num_iter, num_particles,rn_range, benchmark_function):
    # PSO initialisation particle_best (Sp_best), group_best(Sg_best), 
    #   particle(Sp), particle_velocity(Vp)
    Sp = np.zeros((num_particles,2))
    Sp_best = np.zeros((num_particles,2))
    Sg_best = np.zeros(2).reshape(1,2)
    Vp = np.zeros((num_particles,2))
    a, b, c = 0.3, 0.3, 0.3

    output = []
    for i in range(num_particles):
        Sp[i] = np.array([rn()*rn_range[0]+rn_range[1], 
            rn()*rn_range[0]+rn_range[1]])
        # Sp[i] = np.array([rn_range[1], rn_range[1]])
        Sp_best[i] = Sp[i]

        if output == []:
            output = np.array([[Sp[i][0]], [Sp[i][1]], [benchmark_function(Sp[i].reshape(2,1))]])
            Sg_best = Sp[i]
            
        else:
            output = np.hstack((output, np.array([[Sp[i][0]], [Sp[i][1]], [benchmark_function(Sp[i].reshape(2,1))]])))
            if benchmark_function(Sp[i].reshape(2,1)) <= benchmark_function(Sg_best.reshape(2,1)):
                Sg_best = Sp[i]

    # Particle Swarm Optimisation
    for _ in range(num_iter):
        
        for i in range(num_particles):
            
            Vp[i] = a * Vp[i] + b*rn()*Sp_best[i] + c*rn()*Sg_best
            Sp[i] = Sp[i] * Vp[i]
            if benchmark_function(Sp[i].reshape(2,1)) <= benchmark_function(Sp_best[i].reshape(2,1)):
                Sp_best[i] = Sp[i]

            if benchmark_function(Sp[i].reshape(2,1)) <= benchmark_function(Sg_best.reshape(2,1)):
                Sg_best = Sp[i]
                    
            output = np.hstack((output,[[Sp[i][0]],[Sp[i][1]],[benchmark_function(Sp[i].reshape(2,1))]]))
    return output

#Parameters
num_iter = 250
step_size = 0.001
num_particles = 5
#random number range [0] for multiplication and [1] for substraction
rn = np.random.random
rn_range = [10,-5]

#Benchmark on rosenbrock
ax = plotSurface(rn_range, rosenbrock)

GD_plot = gradientDescent(num_iter, rn_range, rosenbrock, rosenbrock_grad)
ax.scatter(GD_plot[0], GD_plot[1], GD_plot[2], c='b', marker='x')

PSO_plot = particleSwarmOptimisation(num_iter, num_particles, rn_range, rosenbrock)
ax.scatter(PSO_plot[0], PSO_plot[1], PSO_plot[2], c='r', marker='x')
plt.show()
fmt = '{:<12}{:<25}{}'
print(fmt.format("Iteration", "Gradient descent ", "Particle Swarm Optimisation"))
[print(fmt.format(i, gd, pso)) for i, (gd, pso) in enumerate(zip(GD_plot[2], PSO_plot[2]))]


#Benchmark on rastrigin
ax = plotSurface(rn_range, rastrigin)

GD_plot = gradientDescent(num_iter, rn_range, rastrigin, rastrigin_grad)
ax.scatter(GD_plot[0], GD_plot[1], GD_plot[2], c='b', marker='x')

PSO_plot = particleSwarmOptimisation(num_iter, num_particles, rn_range, rastrigin)
ax.scatter(PSO_plot[0], PSO_plot[1], PSO_plot[2], c='r', marker='x')

plt.show()

fmt = '{:<12}{:<25}{}'
print(fmt.format("Iteration", "Gradient descent ", "Particle Swarm Optimisation"))
[print(fmt.format(i, gd, pso)) for i, (gd, pso) in enumerate(zip(GD_plot[2], PSO_plot[2]))]
