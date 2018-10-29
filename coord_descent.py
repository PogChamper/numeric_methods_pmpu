"""
Minimization of quadratic function via coordinate descent method
F = Ax + b
"""
import numpy as np
import stdstram


# A = np.array  ([[4, 0.5, 0.5],
#                [0.5, 7.2, -0.5],
#                [0.5, -0.5, 9.2]])
# b = np.array([1, -2, 3])


# Starting position
x1 = np.array([1, 0, 0])
xk = x1
grad = lambda x: np.dot(A, x) + b

while True:
    # Random descent vecotor
    q1 = np.zeros(3)
    q1[np.random.randint(0, 3)] = 1
    # Evalute step
    mu = - (np.dot(q1, grad(x1))) / np.dot(q1, np.dot(A, q1))
    # Make new step
    xk = x1
    x1 = x1 + mu * q1
    # Break condition
    if np.linalg.norm(grad(x1)) < 1e-6:
        break


def quadraticf(x):
    return 0.5 * np.dot(np.dot(x, A), x) + np.dot(x, b)

print(x1)
lul = quadraticf(x1)
print(lul+6)
