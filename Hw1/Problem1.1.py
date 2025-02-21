import numpy as np

def f(x):
    return np.exp(-x**3) - x**4 - np.sin(x)

def bisection(a, b, tol=0.00005):
    iterations, operations = 0, 0
    while (b - a) / 2 > tol:
        operations += 2 # 1 sub, 1 div
        c = (a + b) / 2
        operations += 2  # 1 add, 1 div
        if f(c) == 0 or (b - a)/2 < tol:
            break
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iterations += 1
        operations += 18  # 3 function calls, each contains 3 exponent, 2 sub, 1 sine
    return c, iterations, operations

def newton(x0, tol=0.00005):
    iterations, operations = 0, 0
    def df(x):
        return -3*x**2 * np.exp(-x**3) - 4*x**3 - np.cos(x)
    while True:
        fx = f(x0)
        dfx = df(x0)
        operations += 16 # 2 function calls, f(x0) contains 6 flops, df(x0) constains 10 flops  
        x1 = x0 - fx / dfx
        operations += 2 # 1 sub, 1 div
        if abs(x1 - x0) < tol:
            break
        operations += 1 # 1 sub
        x0 = x1
        iterations += 1
    return x1, iterations, operations

def secant(x0, x1, tol=0.00005):
    iterations, operations = 0, 0
    while True:
        fx0, fx1 = f(x0), f(x1)
        operations += 12 # 12 flops for fx0 and fx1
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        operations += 5 # 3 sub, 1 mul, 1 div
        if abs(x2 - x1) < tol:
            break
        operations += 1 # 1 sub
        x0, x1 = x1, x2
        iterations += 1
    return x2, iterations, operations

def monte_carlo(a, b, tol=0.00005, max_iter=100000):
    iterations, operations = 0, 0
    best_x, best_f = None, float('inf')
    for _ in range(max_iter):
        x = a + (b - a) * np.random.rand()
        operations += 3 # 1 add, 1 sub, 1 mul
        fx = abs(f(x))
        operations += 6 # 6 flops for f(x)
        if fx < best_f:
            best_x, best_f = x, fx
        if fx < tol:
            break
        iterations += 1
    return best_x, iterations, operations

def main():
    # Running the methods
    x_bisect, i_bisect, f_bisect = bisection(-1, 1) # a = -1, b = 1
    x_newton, i_newton, f_newton = newton(0) # x0 = 0
    x_secant, i_secant, f_secant = secant(-1, 1) # x0 = -1, x1 = 1
    x_monte, i_monte, f_monte = monte_carlo(0.5, 0.75)

    # Output results
    print(f"Bisection: Root = {x_bisect}, Iterations = {i_bisect}, Operations = {f_bisect}")
    print(f"Newton's Method: Root = {x_newton}, Iterations = {i_newton}, Operations = {f_newton}")
    print(f"Secant Method: Root = {x_secant}, Iterations = {i_secant}, Operations = {f_secant}")
    print(f"Monte Carlo: Root = {x_monte}, Iterations = {i_monte}, Operations = {f_monte}")
    
if __name__ == "__main__":
    main()
    