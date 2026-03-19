# assgn4.py
# Name: Aman Pradhan , Roll number: 2311020
# Solve Ax = b using Cholesky factorization and Jacobi Iterative Method

from mylib import CholeskySolver, JacobiSolver

def read_augmented_matrix(filename):
    # Read augmented matrix [A|b]
    A, b = [], []
    with open(filename, "r") as f:
        for line in f:
            row = list(map(float, line.split()))
            A.append(row[:-1])   # matrix A
            b.append(row[-1])    # vector b
    return A, b

def verify_cholesky(A, L):
    # Check if A = L * L^T
    n = len(A)
    LLT = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += L[i][k] * L[j][k]
            LLT[i][j] = s
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - LLT[i][j]) > 1e-6:
                return "INCORRECT: A != L*L^T"
    return "CORRECT: A = L*L^T"

def verify_jacobi(A, b, solver):
    # Check Jacobi formula for last iteration
    last = solver.history[-1]      # [k, x1, x2, ...]
    prev = solver.history[-2]      # [k-1, x1, x2, ...]
    x_prev = prev[1:]
    x_last = last[1:]

    for i in range(len(A)):
        s = 0.0
        for j in range(len(A)):
            if j != i:
                s += A[i][j] * x_prev[j]
        expected = (b[i] - s) / A[i][i]
        if abs(expected - x_last[i]) > 1e-6:
            return "INCORRECT: Jacobi update formula not satisfied"
    return "CORRECT: Jacobi update formula satisfied"

def write_output(filename, chol_solution, chol_msg, jacobi_solver, jacobi_solution, jacobi_msg):
    # Save both results
    with open(filename, "w") as f:
        # Cholesky
        f.write("\tCholesky Factorization\n")
        f.write("Solution x:\n")
        for val in chol_solution:
            f.write(f"{val:.6f}\n")
        f.write("\nVerification:\n" + chol_msg + "\n\n")

        # Jacobi
        f.write("\tJacobi Iteration Results\n")
        header = "(k)" + "".join([f"     x{i+1}(k)" for i in range(jacobi_solver.n)]) + "\n"
        f.write(header)
        for row in jacobi_solver.history:
            f.write("{:<3d}".format(row[0]))
            for val in row[1:]:
                f.write(" {:10.6f}".format(val))
            f.write("\n")
        f.write("\nFinal Solution:\n")
        for val in jacobi_solution:
            f.write(f"{val:.6f}\n")
        f.write("\nVerification:\n" + jacobi_msg + "\n")

if __name__ == "__main__":
    # Step 1: read A and b
    A, b = read_augmented_matrix("assgn4_agumat.txt")

    # Step 2: Cholesky
    chol_solver = CholeskySolver(A, b)
    L = chol_solver.decompose()
    y = chol_solver.forward(L, b)
    chol_solution = chol_solver.backward([list(row) for row in zip(*L)], y)
    chol_msg = verify_cholesky(A, L)

    # Step 3: Jacobi
    jacobi_solver = JacobiSolver(A, b, tol=1e-6)
    jacobi_solution = jacobi_solver.solve()
    jacobi_msg = verify_jacobi(A, b, jacobi_solver)

    # Step 4: save both
    write_output("output.txt", chol_solution, chol_msg, jacobi_solver, jacobi_solution, jacobi_msg)

    print("Cholesky and Jacobi results stored in output.txt")
