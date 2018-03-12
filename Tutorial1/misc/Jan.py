from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from misc.benchmarks import rastrigin


# works for both vectors and atomic values
# todo extend to more than 2 dimensions

# topology

class ParticleSwarmOptimization:
    def __init__(self, n_p, dims, x_min, x_max, func, seed):
        self.dims = dims
        self.a, self.b, self.c = 0.4, 2, 2  # parameters set according to slide 21 lecture 2
        self.n_p = n_p
        self.v_max = (x_max - x_min) / 10
        np.random.seed(seed)

        self.v = np.random.random((n_p, dims)) * (x_max - x_min) - x_max

        self.s = np.array(np.random.random((n_p, dims)) * (x_max - x_min)) - x_max

        self.func = func
        self.rand = np.random
        self.s_pbest = np.ndarray.copy(self.s)
        self.s_gbest = self.s_pbest[np.argmin(self.func(self.s_pbest))]

    def run(self, n):
        hist = np.array(np.zeros((n, self.n_p, self.dims)))
        for t in range(n):
            R1, R2 = self.rand.random(), self.rand.random()
            # update velocities
            self.v = self.a * self.v + self.b * R1 * (self.s_pbest - self.s) + self.c * R2 * (self.s_gbest - self.s)
            # todo clip v
            self.s = self.s + self.v
            hist[t] = self.s
            # with new positions update s_pbest and s_gbest
            # update global
            mask = np.array([np.argmin(np.stack((self.func(self.s), self.func(self.s_pbest)), 1), 1)]).T
            self.s_pbest = np.choose(mask, [self.s, self.s_pbest])
            self.s_gbest = self.s_pbest[np.argmin(self.func(self.s_pbest))]
        print("after {} iterations {} is global best with value {}".format(n, self.s_gbest, self.func(np.array([self.s_gbest]))))
        return hist

if __name__ == '__main__':

    startTime = datetime.now()
    opt = ParticleSwarmOptimization(20, 2, -5, 5, rastrigin, 10)
    hist = opt.run(200)

    X1 = np.arange(-5.1, 5.1, 0.1)
    X2 = np.arange(-5.1, 5.1, 0.1)
    X, Y = np.meshgrid(X1, X2)

    Z = rastrigin(np.array([X.flatten(), Y.flatten()]).T).reshape((len(X), len(Y)))

    # Plot the surface.
    fig = plt.figure()
    ax = Axes3D(fig)
    # surf = ax.plot_wireframe(X, Y, Z)
    surf = ax.plot_surface(X,Y,Z, cmap=cm.gray, linewidth=0, alpha=0.4, rstride=1, cstride=1, #antialiased=False
                                         )
    hist = hist[[x for x in range(0, len(hist), 100)]].reshape(-1, 2)
    ax.scatter(hist[:,0], hist[:,1], rastrigin(hist), c="g", marker="x", s=50)
    plt.show()
    print(datetime.now() - startTime)