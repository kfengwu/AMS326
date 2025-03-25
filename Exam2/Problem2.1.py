import random
import numpy as np
def f(x):
    return np.exp(x ** 5)  # Approximation of e^(x^5)

def midpoint_rule(a, b, N):
    h = (b - a) / N
    result = sum(f(a + (i + 0.5) * h) for i in range(N))
    return h * result

def simpson_1_3(a, b, N):
    h = (b - a) / (N - 1)
    result = f(a) + f(b) + 4 * sum(f(a + i * h) for i in range(1, N, 2)) + 2 * sum(f(a + i * h) for i in range(2, N-1, 2))
    return (h / 3) * result

def simpson_3_8(a, b, N):
    h = (b - a) / (N - 1)
    result = f(a) + f(b) + 3 * sum(f(a + i * h) for i in range(1, N-1, 3)) + 3 * sum(f(a + i * h) for i in range(2, N-1, 3)) + 2 * sum(f(a + i * h) for i in range(3, N-1, 3))
    return (3 * h / 8) * result

def gaussian_quadrature(a, b, N):
    points = [-0.90618, -0.53847, 0, 0.53847, 0.90618]
    weights = [0.23693, 0.47863, 0.56889, 0.47863, 0.23693]
    result = sum(w * f((b-a)/2 * x + (b+a)/2) for x, w in zip(points, weights))
    return (b-a)/2 * result

def monte_carlo(a, b, N):
    result = sum(f(a + (b - a) * random.random()) for _ in range(N))
    return (b - a) * result / N

a, b = -1, 1  # Integration limits
I_actual = 2.0949681713212

methods = {
    "Midpoint": (midpoint_rule, 100),
    "Simpson 1/3": (simpson_1_3, 101),
    "Simpson 3/8": (simpson_3_8, 101),
    "Gaussian Quadrature": (gaussian_quadrature, 5),
    "Monte Carlo": (monte_carlo, 1000),
}

for name, (method, N) in methods.items():
    integral = method(a, b, N)
    error = abs(I_actual - integral)
    print(f"{name}: Integral = {integral:.10f}, Error = {error:.10f}")
