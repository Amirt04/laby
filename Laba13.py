import numpy as np
import matplotlib.pyplot as plt

# Построение равномерной сетки: точки от 0 до 1 и шаг h = 1/N

def build_grid(N):
    x = np.linspace(0, 1, N+1)
    h = 1 / N
    return x, h

# Аналитическое решение для тестовой задачи
def u_exact(x):
    return np.sin(np.pi * x) / (np.pi**2)


# 1. Метод аппроксимации квадратичного функционала
def method_functional(N):
    x, h = build_grid(N)

    A = np.zeros((N-1, N-1))
    b = np.zeros(N-1)

    for i in range(N-1):
        A[i, i] = 2 / h # главный элемент матрицы
        if i > 0:
            A[i, i-1] = -1 / h  # нижняя диагональ
        if i < N-2:
            A[i, i+1] = -1 / h # верхняя диагональ

        # Правая часть 
        xi = x[i+1]
        b[i] = h * np.sin(np.pi * xi)

    u_inner = np.linalg.solve(A, b)

    u = np.zeros(N+1)
    u[1:N] = u_inner
    return x, u


# 2. Метод дельта-функции — сосредоточенный источник
def method_delta(N, x0=0.5):
    x, h = build_grid(N)

    A = np.zeros((N-1, N-1))
    b = np.zeros(N-1)

    # Определяем номер узла, ближайшего к точке x0
    j0 = int(x0 / h)

    for i in range(N-1):
        # Обычная разностная схема -u'' = [-1, 2, -1]
        A[i, i] = 2
        if i > 0:
            A[i, i-1] = -1
        if i < N-2:
            A[i, i+1] = -1

        # Дельта-функция
        b[i] = h**2 * (1.0 if i+1 == j0 else 0.0)


    u_inner = np.linalg.solve(A, b)

    u = np.zeros(N+1)
    u[1:N] = u_inner
    return x, u


# 3. Метод сумматорных тождеств
def method_integral(N):
    x, h = build_grid(N)

    A = np.zeros((N-1, N-1))  # матрица трехдиагональная
    b = np.zeros(N-1)

    for i in range(1, N):
        # -u_{i-1} + 2u_i - u_{i+1} = h^2 f(x_i)
        A[i-1, i-1] = 2
        if i > 1:
            A[i-1, i-2] = -1
        if i < N-1:
            A[i-1, i] = -1

        # Правая часть h^2 * f(x_i)
        b[i-1] = h**2 * np.sin(np.pi * x[i])

    u_inner = np.linalg.solve(A, b)

    # Добавляем граничные условия
    u = np.zeros(N+1)
    u[1:N] = u_inner
    return x, u

def plot_solution(x, u, label):
    plt.plot(x, u, label=label, marker="o")


def main():
    N = 40

    x1, u1 = method_functional(N)
    x2, u2 = method_delta(N, 0.5)
    x3, u3 = method_integral(N)

    # График для 2 методов + аналитическое решение
    plt.figure(figsize=(10, 6))
    plot_solution(x1, u1, "Функционал")
    plot_solution(x3, u3, "Сумматорные тождества")
    plt.plot(x1, u_exact(x1), label="Аналитическое", linewidth=3)

    plt.title("Методы решения краевой задачи — Лабораторная 13")
    plt.grid()
    plt.legend()
    plt.show()

    # График решения с δ-функцией
    plt.figure(figsize=(10, 5))
    plot_solution(x2, u2, "δ-источник")
    plt.title("Сосредоточенный источник тепла")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
