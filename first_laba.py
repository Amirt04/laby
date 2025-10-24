#from scipy.integrate import quad
import math
import random
import matplotlib
from matplotlib import pyplot as plt
# def function(x):
#     return x**3

# limit_low = int(input("Введите нижнее значение интеграла: "))
# limit_high = int(input("Введите верхнее значение интеграла: "))

# value = quad(function, limit_low, limit_high)
# print(f"Значение интеграла: {value}")

def function(x, j, k, m):
    return pow(1 - x, j) - pow(math.sin(pow(x, m)), k)

x = [i * math.pi / 180 for i in range(0, 360, 15)]

j = random.randint(1, 4)
k = random.randint(1, 4)
m = random.randint(1, 4)
print(f"Используем j={j}, k={k}, m={m}")

fx = [function(i, j, k, m) for i in x]

plt.plot(x, fx)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title(f"f(x) = (1-x)^j - sinh(x^m)^k при j={j}, k={k}, m={m}")
plt.grid(True)
plt.show()




