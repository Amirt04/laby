import math
import random
import matplotlib.pyplot as plt
import numpy as np

j = random.randint(1, 4)
k = random.randint(1, 4)
m = random.randint(1, 4)

print(f"Параметры функции: j={j}, k={k}, m={m}")

def f(x):
    return (1 - x)**j - math.sinh(x**m)**k

def bisection(a, b, tol=1e-6, max_iter=50):
    if f(a) * f(b) >= 0:
        print("Функция не меняет знак на отрезке")
        return None
    for _ in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def secant(x0, x1, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        if abs(f(x1) - f(x0)) < 1e-12:  
            return None
        x2 = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
    return x1

root_bisect = bisection(0, 1)
root_secant = secant(0, 1)

print(f"Корень методом дихотомии: {root_bisect}")
print(f"Корень методом секущих: {root_secant}")


x_vals = np.linspace(0, 1, 400)
y_vals = [(1 - x)**j - math.sinh(x**m)**k for x in x_vals]

plt.plot(x_vals, y_vals, label="f(x)")
if root_bisect is not None:
    plt.plot(root_bisect, f(root_bisect), 'ro', label="Корень (дихотомия)")
if root_secant is not None:
    plt.plot(root_secant, f(root_secant), 'go', label="Корень (секущие)")
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Нахождение корня функции")
plt.legend()
plt.grid(True)
plt.show()
