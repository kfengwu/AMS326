import numpy as np

# Function to integrate
def u(x, y):
    return np.exp(x**6 - y**4)

# Exact gradient norm (not approximate)
def grad_u_norm(x, y):
    return np.exp(x**6 - y**4) * np.sqrt(36 * x**10 + 16 * y**6)

# Uniform Monte Carlo Integration over [-1, 1]^2
def monte_carlo_uniform(N=10**6):
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)
    estimate = 4 * np.mean(u(x, y))  # Area of [-1,1]^2 is 4
    return estimate

# Importance Sampling using Rejection Sampling based on |∇u|
def importance_sampling(N=10**6, grid_size=300):
    # Step 1: Estimate normalization constant Z over a grid
    x_grid = np.linspace(-1, 1, grid_size)
    y_grid = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grad_vals = grad_u_norm(X, Y)
    Z = np.mean(grad_vals) * 4  # mean × area

    # Step 2: Rejection sampling to draw samples from |∇u|/Z
    samples = []
    max_grad = np.max(grad_vals)
    max_attempts = 10 * N
    attempts = 0

    while len(samples) < N and attempts < max_attempts:
        x_try = np.random.uniform(-1, 1)
        y_try = np.random.uniform(-1, 1)
        threshold = np.random.uniform(0, max_grad)
        if grad_u_norm(x_try, y_try) >= threshold:
            samples.append((x_try, y_try))
        attempts += 1

    if len(samples) < N:
        print(f"Warning: Only generated {len(samples)} samples after {attempts} attempts")

    samples = np.array(samples)
    x_imp, y_imp = samples[:, 0], samples[:, 1]

    # Step 3: Compute importance weights
    u_vals = u(x_imp, y_imp)
    p_vals = grad_u_norm(x_imp, y_imp) / Z
    weights = u_vals / p_vals
    return np.mean(weights)


# Run uniform method
estimate1 = monte_carlo_uniform()
error1 = abs(estimate1 - 4.028423)
print(f"Uniform Sampling Estimate:    {estimate1:.8f}, Error: {error1:.8e}")

# Run importance sampling method
estimate2 = importance_sampling()
error2 = abs(estimate2 - 4.028423)
print(f"Importance Sampling Estimate: {estimate2:.8f}, Error: {error2:.8e}")
