import numpy as np
from interpolation import f, evalpol
import matplotlib.pyplot as plt


def main():
    x = np.linspace(-2, 2, 4)
    seg = np.arange(-2, 2, 0.01)
    plt.plot(seg, newton_polynomial(x, f(x), seg))
    plt.show()


def _poly_newton_coefficient(x, y):
    """
    x: list or np array contanining x data points
    y: list or np array contanining y data points
    """

    m = len(x)

    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k-1])/(x[k:m] - x[k-1])

    return a


def newton_polynomial(x_data, y_data, x):
    """
    x_data: data points at x
    y_data: data points at y
    x: evaluation point(s)
    """
    a = _poly_newton_coefficient(x_data, y_data)
    n = len(x_data) - 1 # Degree of polynomial
    p = a[n]
    for k in range(1, n+1):
        p = a[n-k] + (x - x_data[n-k])*p
    return p


if __name__ == "__main__":
    main()
