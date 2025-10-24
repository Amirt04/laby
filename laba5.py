import numpy as np
import matplotlib.pyplot as plt

n = 80
a, b = 0.0, 1.0
x = np.linspace(a, b, n + 1)
h = (b - a) / n

def y(x):
    return x * np.sin(x)

def y1_exact(x):
    return np.sin(x) + x * np.cos(x)

def y2_exact(x):
    return 2.0 * np.cos(x) - x * np.sin(x)

yv  = y(x)
y1t = y1_exact(x)
y2t = y2_exact(x)

y1n = np.empty_like(x)
y1n[0]    = (-3*yv[0] + 4*yv[1] - yv[2]) / (2*h)
y1n[1:-1] = (yv[2:] - yv[:-2]) / (2*h)
y1n[-1]   = (3*yv[-1] - 4*yv[-2] + yv[-3]) / (2*h)

y2n = np.empty_like(x)
y2n[0]    = (2*yv[0] - 5*yv[1] + 4*yv[2] - yv[3]) / (h*h)
y2n[1:-1] = (yv[2:] - 2*yv[1:-1] + yv[:-2]) / (h*h)
y2n[-1]   = (2*yv[-1] - 5*yv[-2] + 4*yv[-3] - yv[-4]) / (h*h)

e1 = np.abs(y1t - y1n)
e2 = np.abs(y2t - y2n)

e1_max = e1.max(); j1_max = int(e1.argmax())
e2_max = e2.max(); j2_max = int(e2.argmax())

rmse1 = float(np.sqrt(np.mean(e1**2)))
rmse2 = float(np.sqrt(np.mean(e2**2)))

print(f"h = {h:.6f}, n = {n}")
print(f"max|err y'| = {e1_max:.6e} at j={j1_max}, x={x[j1_max]:.6f}")
print(f"RMSE y'     = {rmse1:.6e}")
print(f"max|err y''|= {e2_max:.6e} at j={j2_max}, x={x[j2_max]:.6f}")
print(f"RMSE y''    = {rmse2:.6e}")

plt.figure(figsize=(9, 4.5))
plt.plot(x, yv, color='purple', label="y(x) = x·sin(x)")
plt.title("Исходная функция y(x) = x·sin(x)")
plt.xlabel("x"); plt.ylabel("y(x)")
plt.legend(); plt.grid(True)

plt.figure(figsize=(9,4.5))
plt.plot(x, y1t, label="y'(x) — точная")
plt.plot(x, y1n, '--', label="y'(x) — численная")
plt.title("Первая производная для y(x)=x·sin x")
plt.xlabel("x"); plt.ylabel("y'")
plt.legend(); plt.grid(True)

plt.figure(figsize=(9,4.5))
plt.plot(x, y2t, label="y''(x) — точная")
plt.plot(x, y2n, '--', label="y''(x) — численная")
plt.title("Вторая производная для y(x)=x·sin x")
plt.xlabel("x"); plt.ylabel("y''")
plt.legend(); plt.grid(True)

plt.show()
