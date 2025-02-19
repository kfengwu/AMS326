import numpy as np

# Part 1: Polynomial interpolation P4(t) using Lagrange's formula
def lagrange_interpolation(t_values, y_values, t):
    n = len(t_values)
    result = 0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (t - t_values[j]) / (t_values[i] - t_values[j])
        result += term
    return result


# Part 2: Quadratic Fit Q2(t) = a0 + a1 * t + a2 * t^2 using least squares
def quadratic_fit(t_values, y_values):
    # Construct the normal equations A * [a0, a1, a2] = b
    A = np.array([[np.sum(t_values**0), np.sum(t_values**1), np.sum(t_values**2)],
                  [np.sum(t_values**1), np.sum(t_values**2), np.sum(t_values**3)],
                  [np.sum(t_values**2), np.sum(t_values**3), np.sum(t_values**4)]])
    
    b = np.array([np.sum(y_values),
                  np.sum(t_values * y_values),
                  np.sum(t_values**2 * y_values)])
    
    # Solve for the coefficients [a0, a1, a2]
    coeffs = np.linalg.solve(A, b)
    return coeffs


def main():
    # Given data points
    t_values = np.array([1, 2, 3, 4, 5])
    y_values = np.array([412, 407, 397, 398, 417])

    # Compute P4(t) at t = 6
    P4_at_6 = lagrange_interpolation(t_values, y_values, 6)

    # Get the quadratic coefficients
    a0, a1, a2 = quadratic_fit(t_values, y_values)

    # Compute Q2(t) at t = 6
    Q2_at_6 = a0 + a1 * 6 + a2 * 6**2

    # Output the results
    print(f"Polynomial interpolation P4(t) at t=6: {P4_at_6}")
    print(f"Quadratic fit Q2(t) at t=6: {Q2_at_6}")

if __name__ == "__main__":
    main()