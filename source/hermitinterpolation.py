import numpy as np
import matplotlib.pyplot as plt
from interpolation import f, evalpol


def hermvander(x, deg):

    ideg = int(deg)

    x = np.array(x, copy=0, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    v[0] = x*0 + 1
    if ideg > 0:
        x2 = x*2
        v[1] = x2
        for i in range(2, ideg + 1):
            v[i] = (v[i-1]*x2 - v[i-2]*(2*(i - 1)))
    return np.moveaxis(v, 0, -1)



def hermfit(x, y, deg, rcond=None, full=False, w=None):

    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0
    deg = np.asarray(deg)

    if deg.ndim == 0:
        lmax = deg
        order = lmax + 1
        van = hermvander(x, lmax)
    else:
        deg = np.sort(deg)
        lmax = deg[-1]
        order = len(deg)
        van = hermvander(x, lmax)[:, deg]

    # set up the least squares matrices in transposed form
    lhs = van.T
    rhs = y.T


    # set rcond
    if rcond is None:
        rcond = len(x)*np.finfo(x.dtype).eps

    # Determine the norms of the design matrix columns.
    if issubclass(lhs.dtype.type, np.complexfloating):
        scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
    else:
        scl = np.sqrt(np.square(lhs).sum(1))
    scl[scl == 0] = 1

    # Solve the least squares problem.
    c, resids, rank, s = np.linalg.lstsq(lhs.T/scl, rhs.T, rcond)
    c = (c.T/scl).T

    return c


def hermval(x, c, tensor=True):

    c = np.array(c, ndmin=1, copy=0)

    x2 = x*2
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1*(2*(nd - 1))
            c1 = tmp + c1*x2
    return c0 + c1*x2

x = np.linspace(-2, 2, 19)

coef = list((hermfit(x, f(x), 10)))
print(f(x))

s = np.arange(-2, 2, 0.01)
y = [evalpol(r, coef) for r in s]
plt.plot(s, hermval(s, coef))
plt.show()