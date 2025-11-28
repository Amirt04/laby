import numpy as np
import math
import matplotlib.pyplot as plt

# размеры области
Lx = 1.0
Ly = 1.0

Nx = 50
Ny = 50

hx = Lx / Nx
hy = Ly / Ny

# сетки по x и y
x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)

# правая часть f(x,y)
f = np.zeros((Nx+1, Ny+1))
for i in range(Nx+1):
    for j in range(Ny+1):
        f[i, j] = 2 * math.pi**2 * math.sin(math.pi*x[i]) * math.sin(math.pi*y[j])

# начальное приближение
u = np.zeros((Nx+1, Ny+1))

# итерации Зейделя
eps = 1e-6
max_iter = 10000

for k in range(max_iter):
    max_diff = 0.0

    for i in range(1, Nx):
        for j in range(1, Ny):
            u_new = ( (u[i+1,j] + u[i-1,j]) / hx**2 +
                      (u[i,j+1] + u[i,j-1]) / hy**2 -
                      f[i,j] ) / (2/hx**2 + 2/hy**2)

            diff = abs(u_new - u[i,j])
            if diff > max_diff:
                max_diff = diff

            u[i,j] = u_new

    if max_diff < eps:
        print("Сошлось за", k, "итераций")
        break

# точное решение
u_exact = np.zeros((Nx+1, Ny+1))
for i in range(Nx+1):
    for j in range(Ny+1):
        u_exact[i,j] = math.sin(math.pi*x[i]) * math.sin(math.pi*y[j])

# ошибка
error = np.max(np.abs(u - u_exact))
print("Максимальная ошибка:", error)

# график
plt.figure()
plt.imshow(u, origin='lower', cmap='hot', extent=[0, Lx, 0, Ly])
plt.colorbar()
plt.title("Численное решение u(x,y)")
plt.show()

plt.figure()
plt.imshow(u_exact, origin='lower', cmap='hot', extent=[0, Lx, 0, Ly])
plt.colorbar()
plt.title("Точное решение")
plt.show()
