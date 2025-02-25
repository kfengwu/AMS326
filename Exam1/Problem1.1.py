import numpy as np

# Data points
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
t_values = [16 + 31 * i for i in range(12)]
avg_temps = [33, 34, 40, 51, 60, 69, 75, 74, 67, 56, 47, 38]

# Fit a 3rd degree polynomial
def fit_polynomial(t_values, avg_temps):
    coeffs = np.polyfit(t_values, avg_temps, 3)
    poly = np.poly1d(coeffs)
    return poly

# Polynomial fitting
polynomial = fit_polynomial(t_values, avg_temps)
print("Polynomial Coefficients:", polynomial)

# Predictions for June 4 and Dec 25
t_june4 = 16 + 31 * 5 + (4 - 1)  # June 4
print("Temperature on June 4:", polynomial(t_june4))

t_dec25 = 16 + 31 * 11 + (25 - 1)  # Dec 25
print("Temperature on Dec 25:", polynomial(t_dec25))

# Finding days when temperature is 64.89
def find_roots(polynomial, target_temp):
    coeffs = polynomial.coefficients.copy()
    coeffs[-1] -= target_temp
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if np.isreal(r) and 0 <= r <= 365]
    return real_roots

target_temp = 64.89
days = find_roots(polynomial, target_temp)
print("Days when temperature reaches 64.89:", days)
