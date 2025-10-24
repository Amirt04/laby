import numpy as np
import matplotlib.pyplot as plt

def inf_norm(v):
    return np.max(np.abs(v))

def inf_norm_res(A, x, b):
    return inf_norm(A @ x - b)

def lu_doolittle(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    U = np.zeros_like(A, dtype=float)
    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]
        if abs(U[i, i]) < 1e-15:
            raise ZeroDivisionError("Zero pivot")
        for j in range(i, n):
            L[j, i] = (A[j, i] - L[j, :i] @ U[:i, i]) / U[i, i]
    return L, U

def forward_subst(L, b):
    n = L.shape[0]
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y

def back_subst(U, y):
    n = U.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x

def gauss_seidel(A, b, maxit=20000, tol=1e-12):
    n = A.shape[0]
    x = np.zeros(n, dtype=float)
    D = np.diag(A)
    if np.any(np.abs(D) < 1e-15):
        raise ZeroDivisionError("Zero on diagonal")
    residuals = []
    for k in range(maxit):
        for i in range(n):
            s = A[i, :] @ x - A[i, i] * x[i]
            x[i] = (b[i] - s) / A[i, i]
        r = inf_norm_res(A, x, b)
        residuals.append(r)
        if r <= tol:
            break
    return x, residuals

def make_variant4(n, p=1, q=4):
    if not (5 <= n <= 20):
        raise ValueError("n must be in [5..20]")
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    idx = np.arange(1, n + 1, dtype=float)
    for i in range(n):
        A[i, i] = 8.0 * (idx[i] ** (p / 3.0))
        b[i] = 7.0 * (idx[i] ** (p / 3.0))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sgn = 1.0 if ((i + j) % 2 == 0) else -1.0
            denom = (idx[i] ** p) + sgn * (idx[j] ** q)
            A[i, j] = sgn * 1e-2 / denom
    return A, b

def main():
    n, p, q = 10, 1, 4
    A, b = make_variant4(n, p, q)

    L, U = lu_doolittle(A)
    y = forward_subst(L, b)
    x_lu = back_subst(U, y)

    x_gs, residuals = gauss_seidel(A, b, maxit=20000, tol=1e-12)

    np.set_printoptions(precision=10, suppress=False)
    print(f"n = {n}, p = {p}, q = {q}")
    print("x (LU):")
    print(x_lu)
    print("||Ax-b||_inf (LU) =", inf_norm_res(A, x_lu, b))
    print()
    print("x (Seidel):")
    print(x_gs)
    print("||Ax-b||_inf (Seidel) =", inf_norm_res(A, x_gs, b))
    print("iters Seidel =", len(residuals))

    plt.figure(figsize=(7,4))
    plt.plot(residuals, marker='o')
    plt.yscale('log')
    plt.xlabel("k")
    plt.ylabel("||Ax-b||_inf")
    plt.title("Сходимость метода Зейделя (вариант 4, n=%d)" % n)
    plt.grid(True)

    plt.figure(figsize=(7,4))
    diff = np.abs(x_lu - x_gs)
    plt.bar(np.arange(1, n+1), diff)
    plt.xlabel("i")
    plt.ylabel("|x_LU - x_Seidel|")
    plt.title("Сравнение решений LU и Зейделя")
    plt.grid(True, axis='y')

    plt.show()

if __name__ == "__main__":
    main()
