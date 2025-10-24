import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

a, b, k, m = 1, 5, 1, 4
x0, x1 = 0.0, 1.0
INTERVAL_MS = 120
N_EVAL = 1200

def y_true(x):
    return x**k / (a + b*x)**m

def newton2_local(x_eval, x_nodes, y_nodes):
    h = x_nodes[1] - x_nodes[0]
    n_loc = len(x_nodes) - 1
    P = np.empty_like(x_eval, dtype=float)
    for idx, x in enumerate(x_eval):
        j = np.searchsorted(x_nodes, x) - 1
        if j < 0: j = 0
        if j > n_loc - 2: j = n_loc - 2
        y0, y1, y2 = y_nodes[j], y_nodes[j+1], y_nodes[j+2]
        f1 = y1 - y0
        f2 = y2 - 2.0*y1 + y0
        t0 = (x - x_nodes[j]) / h
        t1 = (x - x_nodes[j+1]) / h
        P[idx] = y0 + t0 * f1 + t0 * t1 * f2
    return P

for n in range(20, 101):
    x_nodes = np.linspace(x0, x1, n + 1)
    y_nodes = y_true(x_nodes)
    x_half = (x_nodes[:-1] + x_nodes[1:]) / 2.0
    P_half = newton2_local(x_half, x_nodes, y_nodes)
    true_half = y_true(x_half)
    errors = true_half - P_half
    eps_max = np.max(np.abs(errors))
    mse = np.mean(errors**2)
    eps_rms = np.sqrt(mse)
    print(f"n={n:3d}  ε_max={eps_max:.6e}  MSE={mse:.6e}  ε_m={eps_rms:.6e}")

n = 20
x_nodes = np.linspace(x0, x1, n + 1)
y_nodes = y_true(x_nodes)
x_half = (x_nodes[:-1] + x_nodes[1:]) / 2.0
P_half = newton2_local(x_half, x_nodes, y_nodes)
x_eval = np.linspace(x0, x1, N_EVAL)
P_eval = newton2_local(x_eval, x_nodes, y_nodes)
y_true_eval = y_true(x_eval)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_eval, y_true_eval, 'g--', lw=1.5, label='y(x)')
ax.plot(x_nodes, y_nodes, 'bo', ms=5, label='узлы')
interp_points, = ax.plot([], [], 'rx', ms=6, mew=1.5, label='P(x_{j+1/2})')
interp_line, = ax.plot([], [], 'r-', lw=2.0, label='Интерполяция')
head_point, = ax.plot([], [], 'ro', ms=5)
y_min = min(P_eval.min(), y_true_eval.min(), y_nodes.min())
y_max = max(P_eval.max(), y_true_eval.max(), y_nodes.max())
pad = 0.06 * (y_max - y_min if y_max > y_min else 1.0)
ax.set_xlim(x0, x1)
ax.set_ylim(y_min - pad, y_max + pad)
ax.set_title('Интерполяция Ньютона 2-го порядка')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
ax.legend(loc='best')
num_frames = len(x_half)

def update(frame_idx):
    # показываем первые frame_idx полуцелых точек
    interp_points.set_data(x_half[:frame_idx], P_half[:frame_idx])

    # куда дорисовывать красную линию (доля по X от 0 до 1)
    last_x = x0 + (x1 - x0) * (frame_idx / num_frames)
    i_eval = int(np.searchsorted(x_eval, last_x))
    if i_eval < 1: i_eval = 1  # защита от пустого среза

    # дорисовываем интерполяцию до индекса i_eval
    interp_line.set_data(x_eval[:i_eval], P_eval[:i_eval])

    # "головка" линии — последняя отображаемая точка
    head_point.set_data([x_eval[i_eval-1]], [P_eval[i_eval-1]])

    # возвращаем объекты, которые перерисовываются
    return interp_points, interp_line, head_point


ani = FuncAnimation(             # создаём анимацию
    fig,                         # объект фигуры
    update,                      # функция, вызываемая каждый кадр
    frames=range(1, num_frames + 1),  # количество кадров = число полуцелых точек
    interval=INTERVAL_MS,        # задержка между кадрами (мс)
    blit=False,                  # перерисовывать всё (надёжнее для Tkinter)
    repeat=False                 # не зацикливать анимацию
)
plt.show()                       

