import numpy as np

# Method to generate matrices A and B uniformly
def generate_matrices(n):
    A = np.random.uniform(-1, 1, (n, n))
    B = np.random.uniform(-1, 1, (n, n))
    return A, B

# Method to summarize and print the matrices
def summarize_matrix(name, M, preview_size=5):
    print(f"\n{name} Summary:")
    print(f"Shape: {M.shape}")
    print(f"Min: {M.min():.4f}, Max: {M.max():.4f}")
    print(f"Mean: {M.mean():.4f}, Std: {M.std():.4f}")
    print(f"Top-left {preview_size}x{preview_size} block:") # Display only the top-left 5 rows
    print(M[:preview_size, :preview_size])

# Method to compute C by Strassen alforithmm of 2 levels
def strassen_2level(A, B):
    def standard_multiply(X, Y):
        return X @ Y

    def strassen_level(X, Y):
        n = X.shape[0]
        mid = n // 2
        A11, A12 = X[:mid, :mid], X[:mid, mid:]
        A21, A22 = X[mid:, :mid], X[mid:, mid:]
        B11, B12 = Y[:mid, :mid], Y[:mid, mid:]
        B21, B22 = Y[mid:, :mid], Y[mid:, mid:]

        M1 = standard_multiply(A11 + A22, B11 + B22)
        M2 = standard_multiply(A21 + A22, B11)
        M3 = standard_multiply(A11, B12 - B22)
        M4 = standard_multiply(A22, B21 - B11)
        M5 = standard_multiply(A11 + A12, B22)
        M6 = standard_multiply(A21 - A11, B11 + B12)
        M7 = standard_multiply(A12 - A22, B21 + B22)

        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6

        return np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

    def strassen_2(X, Y):
        n = X.shape[0]
        if n <= 256:
            return standard_multiply(X, Y)

        mid = n // 2
        A11, A12 = X[:mid, :mid], X[:mid, mid:]
        A21, A22 = X[mid:, :mid], X[mid:, mid:]
        B11, B12 = Y[:mid, :mid], Y[:mid, mid:]
        B21, B22 = Y[mid:, :mid], Y[mid:, mid:]

        M1 = strassen_level(A11 + A22, B11 + B22)
        M2 = strassen_level(A21 + A22, B11)
        M3 = strassen_level(A11, B12 - B22)
        M4 = strassen_level(A22, B21 - B11)
        M5 = strassen_level(A11 + A12, B22)
        M6 = strassen_level(A21 - A11, B11 + B12)
        M7 = strassen_level(A12 - A22, B21 + B22)

        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6

        return np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

    return strassen_2(A, B)
# Method to calculate invert matrix of C
def invert_matrix(C):
    try:
        C_inv = np.linalg.inv(C)
        return C_inv
    except np.linalg.LinAlgError:
        print("Matrix is singular and cannot be inverted.")
        return None

n = 2**10

A, B = generate_matrices(n)
summarize_matrix("Matrix A", A)
summarize_matrix("Matrix B", B)

C = strassen_2level(A, B)
summarize_matrix("Matrix C = A x B", C)

C_inv = invert_matrix(C)
if C_inv is not None:
    summarize_matrix("Matrix C^-1", C_inv)
else:
    print("C^-1 computation failed.")

