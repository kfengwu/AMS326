import numpy as np
import matplotlib.pyplot as plt

# --- Part 1: Solve IVP numerically using implicit method ---

def F(x_prime, x_old, t_i):
    return np.exp(-x_prime) - x_prime - x_old**3 + 3 * np.exp(-t_i**3)

def dFdx(x_prime):
    return -np.exp(-x_prime) - 1

# Parameters
N = 10**4
t = np.linspace(0, 5, N)
h = t[1] - t[0]
x = np.zeros(N)
x[0] = 1  # Initial condition

# Implicit Euler method with Newton-Raphson
tol = 1e-8
for i in range(1, N):
    x_old = x[i-1]
    t_i = t[i-1]
    x_prime_guess = 0.0

    for _ in range(10):
        f_val = F(x_prime_guess, x_old, t_i)
        df_val = dFdx(x_prime_guess)
        if abs(f_val) < tol:
            break
        x_prime_guess -= f_val / df_val
    else:
        print(f"Warning: Newton-Raphson did not converge at step {i}, t = {t_i:.5f}")

    x[i] = x_old + h * x_prime_guess

# Plot the numerical solution
plt.figure(figsize=(10, 6))
plt.plot(t, x, label='Numerical solution $x(t)$')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Solution of the IVP')
plt.grid(True)
plt.legend()
plt.show()

# --- Part 2: Polynomial interpolation using 6 points ---

sample_t = np.array([0, 1, 2, 3, 4, 5])
sample_x = np.interp(sample_t, t, x)

# Fit polynomial of degree 5
poly_coeffs = np.polyfit(sample_t, sample_x, 5)
P = np.poly1d(poly_coeffs)

# Residuals and R²
poly_residuals = sample_x - P(sample_t)
r2_poly = 1 - np.sum(poly_residuals**2) / np.sum((sample_x - np.mean(sample_x))**2)

# Plot
t_dense = np.linspace(0, 5, 1000)
plt.figure(figsize=(10, 6))
plt.plot(t, x, label='Original solution')
plt.plot(t_dense, P(t_dense), 'r--', label=f'Polynomial Interpolation $P_5(t)$\n$R^2 = {r2_poly:.4f}$')
plt.scatter(sample_t, sample_x, color='black')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Polynomial Interpolation')
plt.grid(True)
plt.legend()
plt.show()

# Report coefficients
print("Polynomial Coefficients (highest degree to constant):")
print(poly_coeffs)

# --- Part 3: Manual fit to x(t) = 1 + α * t * exp(-β * t) ---

# grid search to estimate α and β
def x_fit_model(t, alpha, beta):
    return 1 + alpha * t * np.exp(-beta * t)

alpha_vals = np.linspace(0.1, 5.0, 100)
beta_vals = np.linspace(0.1, 5.0, 100)

best_alpha, best_beta = 0, 0
min_error = float('inf')

for alpha in alpha_vals:
    for beta in beta_vals:
        prediction = x_fit_model(sample_t, alpha, beta)
        error = np.sum((sample_x - prediction)**2)
        if error < min_error:
            min_error = error
            best_alpha, best_beta = alpha, beta

# Compute R²
best_fit = x_fit_model(sample_t, best_alpha, best_beta)
r2_fit = 1 - np.sum((sample_x - best_fit)**2) / np.sum((sample_x - np.mean(sample_x))**2)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, x, label='Original solution')
plt.plot(t_dense, x_fit_model(t_dense, best_alpha, best_beta), 'g--',
         label=fr'$x_{{\mathrm{{Fit}}}}(t) = 1 + \alpha t e^{{-\beta t}}$' '\n'
               fr'$\alpha = {best_alpha:.4f}$, $\beta = {best_beta:.4f}$, $R^2 = {r2_fit:.4f}$')
plt.scatter(sample_t, sample_x, color='black')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('t-exp Fit to the Solution')
plt.grid(True)
plt.legend()
plt.show()

# Report best-fit parameters
print(f"Fitted Parameters:\n alpha = {best_alpha:.4f}\n beta = {best_beta:.4f}")
