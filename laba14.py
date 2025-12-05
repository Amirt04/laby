import numpy as np
import matplotlib.pyplot as plt


# ================================
# Параметры задачи
# ================================
L = 1.0
T = 1.0
Nx = 50
Nt = 2000

hx = L / Nx
ht = T / Nt

x = np.linspace(0, L, Nx+1)
t = np.linspace(0, T, Nt+1)


# ================================
# Прямая задача u_t = u_xx + f(x)
# ================================
def solve_direct(f):
    u = np.zeros((Nt+1, Nx+1))

    # Явная схема
    for n in range(Nt):
        for i in range(1, Nx):
            u[n+1, i] = (
                u[n, i]
                + ht * ( (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / hx**2 + f[i] )
            )
    return u


# ================================
# Сопряжённая задача: -psi_t = psi_xx
# конечное условие psi(x,T) = u(x,T) - u_T(x)
# ================================
def solve_adjoint(r):
    psi = np.zeros((Nt+1, Nx+1))
    psi[Nt, :] = r

    # Решаем назад по времени
    for n in reversed(range(Nt)):
        for i in range(1, Nx):
            psi[n, i] = (
                psi[n+1, i]
                - ht * ( (psi[n+1, i+1] - 2*psi[n+1, i] + psi[n+1, i-1]) / hx**2 )
            )
    return psi


# ================================
# Вычисляем градиент g(x)
# g(x) = ∫ psi(x,t) dt
# ================================
def compute_gradient(psi):
    g = np.trapz(psi, dx=ht, axis=0)
    return g


# ================================
# Истинная функция f(x) и данные u_T
# ================================
def true_f(x):
    return np.sin(np.pi * x)


# Решаем прямую задачу для истинного f и берём u_T(x)
u_true = solve_direct(true_f(x))
uT = u_true[Nt, :]


# ================================
# Итерационный процесс восстановления f(x)
# ================================
def inverse_problem(iterations=10, alpha=0.01):
    f = np.zeros_like(x)  # начальное приближение

    for k in range(iterations):
        u = solve_direct(f)
        r = u[Nt, :] - uT
        psi = solve_adjoint(r)
        g = compute_gradient(psi)

        f = f - alpha * g

        print(f"Итерация {k+1}: норма ошибки = {np.linalg.norm(r):.6f}")

    return f


# ================================
# Запуск
# ================================
f_rec = inverse_problem(iterations=10, alpha=0.01)


# ================================
# Графики
# ================================
plt.figure(figsize=(10, 5))
plt.plot(x, true_f(x), label="Истинное f(x)", linewidth=3)
plt.plot(x, f_rec, label="Восстановленное f(x)")
plt.grid()
plt.legend()
plt.title("Решение обратной задачи 2 (Лабораторная №14)")
plt.show()
