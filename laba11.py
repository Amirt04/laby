import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

a = 1.0          # скорость
L = 1.0          # длина области
N = 100          # число узлов
h = L / N        # шаг по x

c = 0.9          # число Куранта 
tau = c * h / a  # шаг по времени

T = 1.0 # время
kmax = int(T / tau)  # число шагов по времени

def u_exact(x, t):
    return math.sin(math.pi * x) * math.cos(math.pi * t)  # точное решение

x = [j * h for j in range(N + 1)]  # сетка по x

u_prev = [0.0] * (N + 1)  # слой n-1
u_curr = [0.0] * (N + 1)  # слой n
u_next = [0.0] * (N + 1)  # слой n+1

# начальное условие u(x,0)
for j in range(N + 1):
    u_prev[j] = u_exact(x[j], 0.0)

# слой t = tau
for j in range(N + 1):
    u_curr[j] = u_exact(x[j], tau)

# граничные условия
u_prev[0] = u_prev[N] = 0.0
u_curr[0] = u_curr[N] = 0.0

frames = []  # кадры для анимации

# основная схема
for n in range(1, kmax):
    for j in range(1, N):
        u_next[j] = (2*u_curr[j]
                     - u_prev[j]
                     + c*c*(u_curr[j+1] - 2*u_curr[j] + u_curr[j-1])) # трёхслойная разностная схема, волнового уравнения
    u_next[0] = u_next[N] = 0.0  # граница

    frames.append(u_curr.copy()) 

    u_prev, u_curr, u_next = u_curr, u_next, u_prev  


t_final = kmax * tau

# оценка ошибки
max_err = 0.0
for j in range(N + 1):
    exact = u_exact(x[j], t_final)
    err = abs(u_curr[j] - exact)
    if err > max_err:
        max_err = err

print("Параметры сетки:")
print("N =", N)
print("h =", h)
print("tau =", tau)
print("kmax =", kmax)
print("t_final =", t_final)
print()
print("Макс. ошибка =", max_err)
print()
print("Несколько точек:")
for j in range(0, N + 1, 20):
    exact = u_exact(x[j], t_final)
    print(f"x={x[j]:.3f}, num={u_curr[j]:.6f}, exact={exact:.6f}")


y_exact = [u_exact(xj, t_final) for xj in x]

plt.plot(x, u_curr, label="Численное")
plt.plot(x, y_exact, "--", label="Точное")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Сравнение численного и точного решений")
plt.legend()
plt.grid(True)
plt.show()

fig, ax = plt.subplots()
line, = ax.plot(x, frames[0])
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_title("Колебание струны")

def update(k):
    line.set_ydata(frames[k])
    return line,

anim = FuncAnimation(fig, update, frames=len(frames), interval=30)
plt.show()
