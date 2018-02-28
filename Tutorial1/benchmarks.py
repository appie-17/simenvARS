import numpy as np
from math import pi

def rosenbrock(X):
    _a = 0
    _b = 10
    x = X[:,0]
    y = X[:,1]
    return pow(_a - y, 2) + _b * pow(y - x**2, 2)

def rastrigin(X):
    n = X.shape[1]
    return 10 * n + np.sum(np.array([X[:, i] ** 2 - 10 * np.cos(2 * pi * X[:, i]) for i in range(n)]).T, 1)


if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    n_p = 4
    dims = 2
    data = np.random.random((n_p, dims))


    X1 = np.arange(-5.1, 5.1, 0.1)
    X2 = np.arange(-5.1, 5.1, 0.1)
    X, Y = np.meshgrid(X1, X2)

    Z = rastrigin(np.array([X.flatten(), Y.flatten()])).reshape((len(X), len(Y)))

    # Plot the surface.
    fig = plt.figure()
    ax = Axes3D(fig)
    # surf = ax.plot_wireframe(X, Y, Z)
    surf = ax.plot_surface(X,Y,Z, cmap=cm.jet, linewidth=0, alpha=1, rstride=1, cstride=1, #antialiased=False
                        )
    plt.show()