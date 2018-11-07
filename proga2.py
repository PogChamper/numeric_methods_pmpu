import numpy as np
a = np.array  ([[4, 0.5, 0.5],
               [0.5, 7.2, -0.5],
               [0.5, -0.5, 9.2]])

b = np.array([1, -2, 3])
x1 = np.array([1, 0, 0])
xk = x1;

while True:
    q1 = np.zeros(3)
    q1[np.random.randint(0, 3)] = 1
    tt = np.dot(a, x1) + b
    e = np.dot(a, q1)
    n = - (np.dot(q1, tt)) / np.dot(q1, e)
    xk = x1
    x1 = x1 + n * q1
    if np.linalg.norm(tt) < 1e-6:
        break

def r(xx):
    return 0.5 * np.dot(np.dot(xx, a), xx) + np.dot(xx, b)

print(x1)
lul = r(x1)
print(lul+6)