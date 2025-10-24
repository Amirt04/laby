import numpy as np
import matplotlib.pyplot as plt

a, b = 0.0, 1.0     
n = 20            
q = 3               

x = np.linspace(a, b, n + 1)
y = 1.0 / (1.0 + x**q)

Sx = np.array([np.sum(x**k) for k in range(5)], float)
Sy = np.array([np.sum(y * x**k) for k in range(3)], float)

A = np.array([
    [Sx[0], Sx[1], Sx[2]],
    [Sx[1], Sx[2], Sx[3]],
    [Sx[2], Sx[3], Sx[4]]
], float)
b_vec = Sy

c0, c1, c2 = np.linalg.solve(A, b_vec)

phi = c0 + c1 * x + c2 * x**2
eps = y - phi

eps_max = np.max(np.abs(eps))
eps_rmse = np.sqrt(np.mean(eps**2))

print(f"Параметры: q = {q}, n = {n}")
print(f"Коэффициенты полинома: c0 = {c0:.8f}, c1 = {c1:.8f}, c2 = {c2:.8f}")
print(f"Максимальная погрешность εmax = {eps_max:.6e}")
print(f"Среднеквадратичная погрешность εm = {eps_rmse:.6e}")

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'bo-', label='y(x) = 1 / (1 + x^q)')
plt.plot(x, phi, 'r--', label='φ(x) — аппроксимация (МНК, степень 2)')
plt.xlabel('x')
plt.ylabel('значение функции')
plt.title(f'Аппроксимация функции y(x) = 1/(1+x^{q}) (вариант 13)')
plt.legend()
plt.grid(True)
plt.show()
