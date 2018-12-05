import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
# from mpmath import factorial

def f(x):
    return x**3 + np.exp(x)
    # return np.log(x ** 2) + x ** 3


def integral_f(x):
    return 8 * x - (x ** 4) / 4 + (x ** 7) / 7 + \
    (x * (-8 + x ** 3) * np.log(x ** 2)) / 2 + \
    (x * np.log(x ** 2) ** 2)


def L2f(a,b):
    return integral_f(b) - integral_f(a)

s = np.arange(-1, 1, 0.0001)

plt.subplot(311)
plt.plot(s, f(s), label="source")

# EUCLID SPACE
x = np.linspace(-0.95, 1, 5)
y = f(x)

Q = np.array(list([1, t, t**2, t**3] for t in x))
print(Q)
H = np.dot(Q.transpose(), Q)
b = np.dot(Q.transpose(), np.array(y).transpose())
print(H)
print(b)
a = np.linalg.solve(H, b)

plt.subplot(312)
plt.plot(s, np.polyval(list(reversed(a)), s))
plt.plot(x, f(x), '*')
print(a)

# L2 SPACE

x = sp.Symbol('x', real=True)
# fun = sp.log(x ** 2) + x ** 3
fun = x**3 + sp.exp(x)
c = [sp.integrate(fun * sp.legendre(i, x), (x, -1, 1)) /
     sp.integrate(sp.legendre(i, x) ** 2, (x, -1, 1))
     for i in range(4)]


pol = sum([c[i] * sp.legendre(i, x) for i in range(4)])
print(c)
print(pol)

plt.subplot(313)
f = sp.lambdify(x, pol, "numpy")
plt.plot(s, f(s))
plt.show()


