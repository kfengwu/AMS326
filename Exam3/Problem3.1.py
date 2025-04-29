import numpy as np

def u(x, y):
    return np.exp(-(x**6 + y**6))

def monte_carlo_uniform(N=10**6):
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)
    integrand_values = u(x, y)
    estimate = 4 * np.mean(integrand_values)  # Area of [-1,1]^2 is 4
    return estimate

def grad_u_norm(x, y):
    return 6 * np.exp(-(x**6 + y**6)) * np.sqrt(x**10 + y**10)

def importance_sampling_gaussian(N=10**6, sigma=0.4):
    accepted = []
    batch_size = int(N / 0.25)  # Estimate acceptance rate to reduce retries
    while len(accepted) < N:
        x = np.random.normal(0, sigma, batch_size)
        y = np.random.normal(0, sigma, batch_size)
        mask = (-1 <= x) & (x <= 1) & (-1 <= y) & (y <= 1)
        accepted.extend(zip(x[mask], y[mask]))
    samples = np.array(accepted[:N])
    
    x_vals, y_vals = samples[:, 0], samples[:, 1]
    pdf_vals = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x_vals**2 + y_vals**2) / (2 * sigma**2))
    integrand_vals = u(x_vals, y_vals)
    estimate = np.mean(integrand_vals / pdf_vals)
    return estimate


# Run
estimate1 = monte_carlo_uniform()
error1 = abs(estimate1 - 3.156049594)
print(f"Uniform Sampling Estimate: {estimate1:.8f}, Error: {error1:.8e}")

# Run
estimate2 = importance_sampling_gaussian()
error2 = abs(estimate2 - 3.156049594)
print(f"Importance Sampling Estimate: {estimate2:.8f}, Error: {error2:.8e}")


