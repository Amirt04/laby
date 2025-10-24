import numpy as np
import matplotlib.pyplot as plt

a, b, c = 2.0, 1.5, 1.0
lows3 = np.array([-a, -b, -c])
highs3 = np.array([ a,  b,  c])
rng = np.random.default_rng(42)

def rho3(X):
    x1, x2, x3 = X[:,0], X[:,1], X[:,2]
    alpha = np.array([-5.0, 0.0, 5.0])
    beta  = np.array([0.5, 2.0, 4.0])
    return (np.abs(x1 - alpha[0])**beta[0] +
            np.abs(x2 - alpha[1])**beta[1] +
            np.abs(x3 - alpha[2])**beta[2])

def inside_ellipsoid3(X):
    x, y, z = X[:,0], X[:,1], X[:,2]
    return (x/a)**2 + (y/b)**2 + (z/c)**2 <= 1.0 + 0.0*x

def uniform_in_box(N, lows, highs, rng):
    lows, highs = np.asarray(lows), np.asarray(highs)
    return rng.random((N, len(lows))) * (highs - lows) + lows

def monte_carlo_integral(f, inside_G, lows, highs, N, rng):
    X = uniform_in_box(N, lows, highs, rng)
    mask = inside_G(X)
    vals = f(X) * mask.astype(float)
    W = np.prod(highs - lows)
    I_est = W * vals.mean()
    M = int(mask.sum())
    V_est = W * (M / N)
    return I_est, V_est, M, X, mask

N_list = [100, 500, 1000, 5000, 10000]
results = []

print("ЛР-7. Метод Монте-Карло (вариант 13)")
print(f"Эллипсоид: a={a}, b={b}, c={c}")
print("Функция плотности ρ(x) = Σ|x_i - α_i|^{β_i}\n")

for N in N_list:
    I_est, V_est, M, X, mask = monte_carlo_integral(rho3, inside_ellipsoid3, lows3, highs3, N, rng)
    results.append((N, I_est, V_est))
    print(f"N={N:5d}  Q≈{I_est:12.6f}   V≈{V_est:9.6f}   M={M:5d}")

Ns = np.array([r[0] for r in results])
Qs = np.array([r[1] for r in results])
Vs = np.array([r[2] for r in results])

plt.figure(figsize=(8,4))
plt.plot(Ns, Qs, 'o-', label='Q ≈ ∫ρ(x)dV')
plt.title("Сходимость интеграла Монте-Карло")
plt.xlabel("N — число случайных точек")
plt.ylabel("Приближённое значение Q")
plt.grid(True)
plt.legend()

X_vis = uniform_in_box(5000, lows3, highs3, rng)
mask_vis = inside_ellipsoid3(X_vis)
inside_pts = X_vis[mask_vis]
outside_pts = X_vis[~mask_vis]

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(outside_pts[:,0], outside_pts[:,1], outside_pts[:,2], s=5, c='lightgray', alpha=0.3)
ax.scatter(inside_pts[:,0], inside_pts[:,1], inside_pts[:,2], s=5, c='purple', alpha=0.6)
ax.set_title("Точки внутри и вне эллипсоида")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
plt.show()
