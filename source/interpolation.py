"""
Function interpolation
"""
import numpy as np
import matplotlib.pyplot as plt


def main():
    # number of points and [a, b] interval
    N = 10
    a, b = -2, 2
    pol = interp(f, a, b, N)
    x = np.arange(-2, 2, 0.01)

    # plot results
    x_f = np.arange(a, b, 0.01)
    plt.subplot(211)
    plt.plot(x_f, f(x_f))
    y = [evalpol(p, pol) for p in x]

    plt.subplot(212)
    plt.plot(x, y, color="green")
    plt.show()


def f(x):
    return np.tan(0.5*x + 0.2) - x ** 2


def evalpol(x, a):
    sum=0
    for i, ai in enumerate(a):
        sum += ai * x ** (len(a) - i - 1)
    return sum


def interp(func, a, b, N):
    x = np.arange(a, b, (b - a) / N)
    x = x[:N]
    A = np.array([[p ** n for n in reversed(range(0, N))] for p in x])
    B = np.array([func(y) for y in x])
    pol = np.linalg.solve(A, B)
    return pol


if __name__ == "__main__":
    main()


