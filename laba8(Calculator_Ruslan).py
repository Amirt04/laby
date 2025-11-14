def tridiagonal_solve(a, b, c, d):
    n = len(d)
    alpha = [0.0] * n
    beta = [0.0] * n

    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * alpha[i - 1]
        if i < n - 1:
            alpha[i] = -c[i] / denom
        else:
            alpha[i] = 0.0
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denom

    x = [0.0] * n
    x[-1] = beta[-1]

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x


def print_system(a, b, c, d):
    n = len(d)
    print("\nСистема уравнений:\n")
    for i in range(n):
        parts = []
        if i > 0 and a[i] != 0:
            parts.append(f"{a[i]}*x{i}")
        parts.append(f"{b[i]}*x{i+1}")
        if i < n - 1 and c[i] != 0:
            parts.append(f"{c[i]}*x{i+2}")
        left = " + ".join(parts)
        print(f"{left} = {d[i]}")
    print()


def main():
    print("Решение трёхдиагональной СЛАУ методом прогонки")
    n = int(input("Введите размер системы n: "))

    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    d = [0.0] * n

    print("\nВвод коэффициентов для уравнений вида:")
    print("a_i * x_(i-1) + b_i * x_i + c_i * x_(i+1) = d_i")
    print("Обычно a1 = 0, cn = 0 для трёхдиагональной матрицы.\n")

    for i in range(n):
        print(f"Уравнение {i + 1}:")
        a[i] = float(input(f"a{i + 1} = "))
        b[i] = float(input(f"b{i + 1} = "))
        c[i] = float(input(f"c{i + 1} = "))
        d[i] = float(input(f"d{i + 1} = "))
        print()

    print_system(a, b, c, d)

    x = tridiagonal_solve(a, b, c, d)

    print("Решение системы:")
    for i, xi in enumerate(x, 1):
        print(f"x{i} = {xi}")


if __name__ == "__main__":
    main()
 