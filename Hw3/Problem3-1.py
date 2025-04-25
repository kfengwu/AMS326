import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 4444444
w = 1  # line spacing
ds = [i/10 for i in range(1, 11)] + [15/10, 20/10, 30/10]
max_lines = 4  # Allow checking up to 4 line crossings

def simulate_crossings(d, n, w):
    y = np.random.uniform(0, w, n)  # random y-position for disc centers
    low = y - d / 2
    high = y + d / 2
    line_low = np.floor(low)
    line_high = np.floor(high)
    num_crossed_lines = line_high - line_low  # how many lines crossed
    return num_crossed_lines.astype(int)

# Store results
results = {d: np.zeros(max_lines + 1, dtype=int) for d in ds}

# Run simulation
for d in ds:
    counts = simulate_crossings(d, n, w)
    for i in range(0, max_lines + 1):
        results[d][i] = np.sum(counts >= i)

# Normalize to get probabilities
probs = {d: results[d] / n for d in ds}

# Plotting
plt.figure(figsize=(12, 7))
for k in range(0, 5):  # plot P(cross >= k lines) for k = 0 to 4
    plt.plot(ds, [probs[d][k] for d in ds], marker='o', label=f'≥ {k} lines')

plt.title('Probability of a disc crossing ≥ k lines (vs. disc diameter)')
plt.xlabel('Disc Diameter (d)')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
