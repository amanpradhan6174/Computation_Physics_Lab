# ludecomp.py
# Aman Pradhan, RollNo: 2311020
# Problem 1: Verify A = L * U  (Doolittle LU decomposition)
# Problem 2: Solve A x = b using forward-backward substitution
# Input: A in assgn3a_matva.txt, Augmented matrix [A|b] in assgn3a_matvb.txt
# Output: output3.txt

def lu_decomposition(A):
    # Doolittle LU decomposition: returns L and U
    n = len(A)
    L = [[0.0]*n for _ in range(n)]
    U = [[0.0]*n for _ in range(n)]

    for i in range(n):
        # compute U
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - s

        # compute L
        L[i][i] = 1.0
        for j in range(i+1, n):
            s = 0.0
            for k in range(i):
                s += L[j][k] * U[k][i]
            L[j][i] = (A[j][i] - s) / U[i][i]

    return L, U

def multiply(L, U):
    # Multiply two matrices
    n = len(L)
    result = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += L[i][k] * U[k][j]
    return result

def forward(L, b):
    # Solve L y = b
    n = len(L)
    y = [0.0]*n
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i][j] * y[j]
        y[i] = b[i] - s
    return y

def backward(U, y):
    # Solve U x = y
    n = len(U)
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        s = 0.0
        for j in range(i+1, n):
            s += U[i][j] * x[j]
        x[i] = (y[i] - s) / U[i][i]
    return x

def read_matrix(filename):
    M = []
    with open(filename, "r") as f:
        for line in f:
            row = [float(x) for x in line.split()]
            M.append(row)
    return M

def write_matrix(f, name, M):
    f.write(name + "\n")
    for row in M:
        f.write(" ".join(f"{val:.2f}" for val in row) + "\n")
    f.write("\n")

def write_vector(f, name, v):
    f.write(name + "\n")
    f.write(" ".join(f"{val:.4f}" for val in v) + "\n\n")

def main():
    # Problem 1
    A = read_matrix("assgn3a_matva.txt")
    L, U = lu_decomposition(A)
    LU_product = multiply(L, U)

    # Check verification
    ok = True
    for i in range(len(A)):
        for j in range(len(A)):
            if abs(A[i][j] - LU_product[i][j]) > 1e-6:
                ok = False

    # Problem 2
    B = read_matrix("assgn3a_matvb.txt")   # augmented matrix
    n = len(B)
    A2 = [row[:-1] for row in B]           # coefficient matrix
    b = [row[-1] for row in B]             # RHS

    L2, U2 = lu_decomposition(A2)
    y = forward(L2, b)
    x = backward(U2, y)

    # Write results
    with open("output3.txt", "w") as f:
        f.write("Q1: LU Decomposition (Doolittle)\n\n")
        write_matrix(f, "Matrix A:", A)
        write_matrix(f, "Matrix L:", L)
        write_matrix(f, "Matrix U:", U)
        write_matrix(f, "Product L*U:", LU_product)
        if ok:
            f.write("Verification: A = L*U (Verified)\n\n")
        else:
            f.write("Verification: Failed (A != L*U)\n\n")

        f.write("Q2: Solve Ax = b using Augmented Matrix\n\n")
        write_vector(f, "Solution x:", x)

    print("Results are in output3.txt")

if __name__ == "__main__":
    main()
