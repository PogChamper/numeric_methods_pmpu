"""
Spline interpolation
"""
import numpy as np
import matplotlib.pyplot as plt
from interpolation import sectionbreak, f, df


def main():
    # N - number of points
    N = 8
    a, b = -2, 2
    x, splines = cubesplineinterp(f, df, a, b, N)

    # plot spline interp
    for i, val in enumerate(x[:-1]):
        st = np.arange(x[i], x[i+1], 0.01)
        plt.plot(st, Poly(splines[i]).eval(st))
    plt.show()


def cubesplineinterp(f, df, a, b, N):
    # split on sections
    x = sectionbreak(a, b, N)
    step = ((b - a) / N)

    # find m for twice continuous
    M = np.zeros((N, N))
    M[0, 0] = 1
    M[N - 1, N - 1] = 1
    for i, row in enumerate(M[1:-1]):
        i += 1
        row[i - 1] = 1
        row[i] = 4
        row[i + 1] = 1
    b = [df(x[0])] + [(3 / step) * (f(x[i + 1]) - f(x[i - 1]))
                      for i, p in enumerate(x[1:-1])] + [df(x[-1])]
    b = np.array(b)
    m = np.linalg.solve(M, b)

    # find coefficient of poly
    splines = []
    for i, val in enumerate(x[:-1]):
        A = [[x[i] ** 3, x[i] ** 2, x[i], 1]]
        A += [[x[i + 1] ** 3, x[i + 1] ** 2, x[i + 1], 1]]
        A += [[3 * x[i] ** 2, 2 * x[i], 1, 0]]
        A += [[3 * x[i + 1] ** 2, 2 * x[i + 1], 1, 0]]
        A = np.array(A)

        b = [f(x[i]), f(x[i + 1]), m[i], m[i + 1]]
        b = np.array(b)

        splines.append(np.linalg.solve(A, b))
    return [x, splines]


class Poly(list):

    # Poly differentiate
    def d(self):
        l = []
        for i, x in enumerate(self):
            l.append((len(self) - 1 - i) * x)
        return Poly([l.pop()] + l)

    def div(self, n):
        if n >= len(self) or n <= 0:
            return [0]
        else:
            if n == 1:
                return self.d()
            else:
                return (self.div(n-1)).d()

    # Poly evalute in x
    def eval(self, x):
        sum = 0
        for i, ai in enumerate(self):
            sum += ai * x ** (len(self) - i - 1)
        return sum


if __name__ == "__main__":
    main()
