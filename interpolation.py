import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return np.tan(0.5*x+0.2)-x*x


def func2(x):
    return abs(x) * (np.tan(0.5*x+0.2)-x*x)


a = int(input())
#points = np.linspace(-1, 1, a) дефолтное разбиение
points = [np.cos(((2*i - 1) / (2 * a)) * np.pi) for i in range(a)] #чебыш
print (points)
Lagrange = np.poly1d(np.zeros(a))
for i in range(0, a):
    omega = np.poly(np.concatenate((points[0:i], points[i+1:])))
    Li = omega / np.polyval(omega, points[i])
    Lagrange = np.polyadd(Lagrange, func(points[i]) * Li)
coordX = np.linspace(-1, 1, 100)
coordY1 = func(coordX)
coordY2 = np.polyval(Lagrange, coordX)
fig, ax = plt.subplots()
ax.plot(coordX, coordY1, color="blue", label="func1")
ax.plot(coordX, coordY2, color="red", label="interpolation")
plt.show()
print(Lagrange)

