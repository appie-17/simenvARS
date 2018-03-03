from autograd import grad
import numpy as np

def rosenbrock(X):
    # A = np.zeros(X.shape[1])
    # B = 10 * np.ones(X.shape[1])
    return np.square(0 - X[0]) + 10 * np.square((X[1] - (X[0]**2)))

X = np.random.rand(2,1)

print(rosenbrock(X))
gradient = grad(rosenbrock)
print(gradient(X))