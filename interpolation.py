"""
Function interpolation
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    # number of points and [a, b] interval
    N = 5
    a, b = -2, 2
    x = np.arange(a, b, (b - a) / N)
    x = x[:N]
    A = np.array([[p ** n for n in reversed(range(0, N))] for p in x])
    B = np.array([f(y) for y in x])
    pol = np.linalg.solve(A, B)

    # plot results
    x_f = np.arange(a, b, 0.01)
    plt.plot(x_f, f(x_f))
    y = [evalpol(p, pol) for p in x]
    plt.plot(x, y, color="green")
    plt.show()


def f(x):
    return np.tan(0.5*x + 0.2) - x ** 2


def evalpol(x, a):
    sum=0
    for i, ai in enumerate(a):
        sum += ai * x ** (len(a) - i - 1)
    return sum


if __name__ == "main":
    main()


