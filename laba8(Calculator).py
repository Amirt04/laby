import numpy as np

def solve_tridiagonal_slae(A, b):
    n = len(b)
    alpha = np.zeros(n)
    beta = np.zeros(n)

    d0 = A[0][0]
    c0 = A[0][1] if n > 1 else 0

    if d0 == 0:
        return f"Ошибка: Диагональный элемент A[0][0] равен нулю ({d0}). Метод прогонки неприменим."

    alpha[0] = -c0 / d0
    beta[0] = b[0] / d0

    for i in range(1, n):
        ai = A[i][i-1]
        di = A[i][i]
        ci = A[i][i+1] if i < n - 1 else 0
        delta = di + ai * alpha[i-1]
        if delta == 0:
            return f"Ошибка: Прогоночный коэффициент delta на шаге {i} равен нулю. Метод неприменим."
        alpha[i] = -ci / delta
        beta[i] = (b[i] - ai * beta[i-1]) / delta

    x = np.zeros(n)
    x[n-1] = beta[n-1]

    for i in range(n - 2, -1, -1):
        x[i] = beta[i] + alpha[i] * x[i+1]

    return x

def get_input_data():
    while True:
        try:
            n = int(input("Введите размерность системы N (например, 5): "))
            if n <= 1:
                print("Размерность должна быть больше 1. Попробуйте снова.")
                continue
            break
        except ValueError:
            print("Ошибка: Введите целое число.")

    A = np.zeros((n, n))
    b = np.zeros(n)
    
    print("\n--- Ввод коэффициентов Матрицы A (только 3 диагонали) ---")
    
    for i in range(n):
        print(f"\nКоэффициенты для строки {i+1}:")
        if i > 0:
            A[i][i-1] = float(input(f"  Коэффициент a_{i+1},{i} (слева): "))
        A[i][i] = float(input(f"  Коэффициент a_{i+1},{i+1} (диагональный): "))
        if i < n - 1:
            A[i][i+1] = float(input(f"  Коэффициент a_{i+1},{i+2} (справа): "))
        b[i] = float(input(f"  Свободный член b_{i+1}: "))

    return A.tolist(), b.tolist()

print("--- 💻 Калькулятор СЛАУ Методом Прогонки (TDMA) ---")
print("⚠️ Вводите только коэффициенты на главной, верхней и нижней диагоналях.")

A_input, b_input = get_input_data()

print("\n--- Выполняем расчёт ---")
print("Введённая матрица A:\n", np.array(A_input))
print("Введённый вектор B:\n", np.array(b_input))
print("-" * 35)

solution = solve_tridiagonal_slae(A_input, b_input)

if isinstance(solution, str):
    print(f"\n❌ Ошибка при решении: {solution}")
else:
    print("\n✅ Система решена успешно:")
    print("Вектор решений X:")
    print(solution)
    print("\nПроверка (A*X - B) [должно быть близко к 0]:", np.dot(A_input, solution) - b_input)
