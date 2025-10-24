import numpy as np
import math
import matplotlib.pyplot as plt

a, b = 0.1, 3.0

def f(x):
    return np.sinh(np.sqrt(x)) + 2.0*x

def F(x):
    r = np.sqrt(x)
    return 2.0*(r*np.cosh(r) - np.sinh(r)) + x**2

I_true = F(b) - F(a)

def rectangles_midpoint(f, a, b, m):
    h = (b - a) / m
    x_mid = a + (np.arange(m) + 0.5) * h
    y_mid = f(x_mid)
    I = h * np.sum(y_mid)
    return I, x_mid, y_mid, h

def gauss_legendre(f, a, b, n):
    t, A = np.polynomial.legendre.leggauss(n)
    c = (b - a)/2.0
    d = (a + b)/2.0
    vals = f(c*t + d)
    I = c * np.dot(A, vals)
    return I, t, A

print(f"Точное значение: I = {I_true:.10f}\n")

m_list = [10, 20, 40, 80, 100]
print("Метод прямоугольников (средних):")
for m in m_list:
    I_rect, _, _, _ = rectangles_midpoint(f, a, b, m)
    err = abs(I_rect - I_true)
    print(f"  m={m:3d}  I≈{I_rect:.10f}   |ошибка|={err:.3e}")
print()

print("Квадратурная формула Гаусса–Лежандра:")
for n in range(5, 12):
    I_g, t, A = gauss_legendre(f, a, b, n)
    err = abs(I_g - I_true)
    print(f"  c_m={n:2d}  I≈{I_g:.10f}   |ошибка|={err:.3e}")
print("\n(Замечание: t — стандартные узлы на [-1,1], A — соответствующие веса.)")

m_vis = 20
I_rect, x_mid, y_mid, h = rectangles_midpoint(f, a, b, m_vis)
X = np.linspace(a, b, 800)
Y = f(X)

plt.figure(figsize=(9,4.8))
plt.plot(X, Y, label='f(x) = sh(√x) + 2x')
plt.bar(x_mid, y_mid, width=h, alpha=0.3, align='center', edgecolor='black',
        label=f'Прямоугольники (m={m_vis})')
plt.title("Численное интегрирование: метод прямоугольников (средних)")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()
