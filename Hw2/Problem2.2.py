import random
def generate_matrix(N):
    return [[random.uniform(-1, 1) for _ in range(N)] for _ in range(N)]

def generate_vector(N):
    return [1] * N

def gaussian_elimination(A,b):
    N = len(A)
    for i in range(N):
        max_row = max(range(i, N), key=lambda r: abs(A[r][i]))
        A[i], A[max_row] = A[max_row], A[i]
        b[i], b[max_row] = b[max_row], b[i]
        for j in range(i + 1, N):
            factor = A[j][i] / A[i][i]
            for k in range(i,N):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]
        x = [0] * N
        for i in range(N - 1, -1, -1):
            x[i] = (b[i] - sum(A[i][j] * x [j] for j in range(i + 1, N))) / A[i][i]
        return x

def main():
    N = 66
    A = generate_matrix(N)
    b = generate_vector(N)
    X = gaussian_elimination(A,b)

    print("Solution Vector X:")
    print(X)

if __name__ == "__main__":
    main()