import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 100          # Initial x position (miles)
w = 44           # Wind speed (miles per hour)
v0 = 88          # Plane speed relative to the wind (miles per hour)
k = w / v0       # k = w / v0
epsilon = 1e-6   # Small value to avoid division by zero

# Differential equation dy/dx
def f(x, y):
    if abs(x) < epsilon:
        x = epsilon
    ratio = y / x
    return ratio - k * np.sqrt(1 + ratio**2)

# Runge-Kutta 4th Order Method
def runge_kutta_4(f, x0, y0, h, n):
    xs = [x0]
    ys = [y0]
    for _ in range(n):
        x, y = xs[-1], ys[-1]
        k1 = f(x, y)
        k2 = f(x + h/2, y + (h/2)*k1)
        k3 = f(x + h/2, y + (h/2)*k2)
        k4 = f(x + h, y + h*k3)
        y_next = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        x_next = x + h
        xs.append(x_next)
        ys.append(y_next)
    return np.array(xs), np.array(ys)

# Simulation parameters
x0 = a       # Start at x = a
y0 = 0       # y(a) = 0
h = -0.01    # Step size (negative for integrating from a to 0)
n_steps = int(abs((a - 0.1) / h))  # Avoid x = 0 directly

# Compute trajectory
x_vals, y_vals = runge_kutta_4(f, x0, y0, h, n_steps)

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="Plane's Trajectory", color='green')
plt.title("Trajectory of a Plane Approaching the Airport")
plt.xlabel("x (miles)")
plt.ylabel("y (miles)")
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
