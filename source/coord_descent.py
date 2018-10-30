"""
Minimization of quadratic function via coordinate descent method
F = Ax + b
"""
import numpy as np


def main():
    N = 6
    A = np.array([[4, 0.5, 0.5],
                  [0.5, 6 + 0.2*N, -0.5],
                  [0.5, -0.5, 8 + 0.2*N]])
    b = np.array([1, -2, 3])
    C = np.array(N)

    x_min = coord_descent(A, b, np.array([1, 0, 0]))
    f_min = quadraticf(x_min, A, b, C)
    print("Via coordinate method:")
    print("Minimum X: %s\nFunction min value: %s" % (x_min, f_min))

    x_min = mngs(A, b, np.array([1, 0, 0]))
    f_min = quadraticf(x_min, A, b, C)
    print("Via mngs:")
    print("Minimum X: %s\nFunction min value: %s" % (x_min, f_min))





def coord_descent(A, b, x1):
    # Starting position
    xk = x1
    grad = lambda x: np.dot(A, x) + b

    while True:
        # Random descent vecotor
        e = np.zeros(3)
        e[np.random.randint(0, 3)] = 1
        # Evalute step
        mu = - (np.dot(e, grad(x1))) / np.dot(e, np.dot(A, e))
        # Make new step
        xk = x1
        x1 = x1 + mu * e
        # Break condition
        if np.linalg.norm(grad(x1)) < 1e-6:
            break

    return x1


def mngs(A, b, x1):
    # Starting position
    xk = x1
    grad = lambda x: np.dot(A, x) + b

    while True:
        # Random descent vecotor
        e = grad(x1)
        # Evalute step
        mu = - (np.linalg.norm(grad(x1) ** 2)) / np.dot(np.dot(grad(x1), A),
                                                        grad(x1))
        # Make new step
        xk = x1
        x1 = x1 + mu * e
        # Break condition
        if np.linalg.norm(grad(x1)) < 1e-6:
            break

    return x1


# Evaluate quadratic function
def quadraticf(x, A, b, C):
    return 0.5 * np.dot(np.dot(x, A), x) + np.dot(x, b) + C


if __name__ == "__main__":
    main()
