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
def least_squares(t_values, y_values):
    n = len(t_values)
    A = [[n, sum(t_values), sum(t**2 for t in t_values)],
         [sum(t_values), sum(t**2 for t in t_values), sum(t**3 for t in t_values)],
         [sum(t**2 for t in t_values), sum(t**3 for t in t_values), sum(t**4 for t in t_values)]]
    
    B = [sum(y_values), sum(t_values[i] * y_values[i] for i in range(n)), sum(t_values[i]**2 * y_values[i] for i in range(n))]
    
    # Solving Ax = B using Gaussian elimination
    n = len(B)
    for i in range(n):
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            B[j] -= factor * B[i]
    
    # Back substitution
    coeffs = np.zeros(n)
    for i in range(n-1, -1, -1):
        coeffs[i] = (B[i] - sum(A[i][j] * coeffs[j] for j in range(i+1, n))) / A[i][i]
    
    return coeffs  # a0, a1, a2

def quadratic_fit(coeffs, t):
    a0, a1, a2 = coeffs
    return a0 + a1 * t + a2 * t**2


def main():
    # Given data points
    t_values = np.array([1, 2, 3, 4, 5])
    y_values = np.array([412, 407, 397, 398, 417])

    # Compute P4(t) at t = 6
    P4_at_6 = lagrange_interpolation(t_values, y_values, 6)

    # Get the quadratic coefficients
    coeffs = least_squares(t_values, y_values)

    # Compute Q2(t) at t = 6
    Q2_at_6 = quadratic_fit(coeffs, 6)

    # Output the results
    print(f"Polynomial interpolation P4(t) at t=6: {P4_at_6}")
    print(f"Quadratic fit Q2(t) at t=6: {Q2_at_6}")

if __name__ == "__main__":
    main()