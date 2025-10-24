import numpy as np

def inf_norm(v):
    return float(np.max(np.abs(v)))

def power_method(A, eps=1e-8, kmax=5000, seed=123):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    x /= inf_norm(x)
    lam_prev = None
    for k in range(1, kmax + 1):
        y = A @ x
        lam = float((x @ y) / (x @ x))
        ny = inf_norm(y)
        if ny == 0.0:
            return 0.0, x, k - 1
        x_new = y / ny
        if (lam_prev is not None and abs(lam - lam_prev) <= eps) or inf_norm(x_new - x) <= eps:
            return lam, x_new, k
        x, lam_prev = x_new, lam
    return lam_prev if lam_prev is not None else 0.0, x, kmax

def build_matrix_variant3(n=10, p=1, q=4, b=0.05):
    i = np.arange(1, n + 1, dtype=float)
    A = np.zeros((n, n), dtype=float)
    for ii in range(n):
        A[ii, ii] = 10.0 * (i[ii] ** (p / 2.0))
    for ii in range(n):
        for jj in range(ii + 1, n):
            val = b * ((i[ii] / (i[jj] ** p) + i[jj] / (i[ii] ** p)) ** (1.0 / q))
            A[ii, jj] = val
            A[jj, ii] = val
    return A

def main():
    n, p, q, b = 10, 1, 4, 0.05  # n∈[5..10], p,q∈[1..4], b∈[0.01..0.1]
    A = build_matrix_variant3(n, p, q, b)
    lam, x, iters = power_method(A, eps=1e-8, kmax=5000, seed=123)
    resid = np.linalg.norm(A @ x - lam * x, ord=np.inf)
    np.set_printoptions(precision=10, suppress=False)
    print(f"Вариант 3: n={n}, p={p}, q={q}, b={b}")
    print(f"λ_max ≈ {lam:.10f}, итераций = {iters}")
    print(f"||Ax - λx||_inf = {resid:.3e}")

if __name__ == "__main__":
    main()
