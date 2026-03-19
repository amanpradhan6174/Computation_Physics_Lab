# Assignment-5
# Name - Aman Pradhan
# Roll No. - 2311020
# Prob 1 - Cholesky and Gauss-Seidel
# Prob 2 - Jacobi and Gauss-Seidel (with rearranging to make diagonally dominant)

from itertools import permutations
from mylib import CholeskySolver, JacobiSolver, GaussSeidelSolver   # import solvers

# Read augmented matrix [A|b] from file
def read_augmented_matrix(filename):
    A, b = [], []
    with open(filename, "r") as f:
        for line in f:
            row = list(map(float, line.split()))
            A.append(row[:-1])   # matrix part
            b.append(row[-1])    # RHS vector
    return A, b

# Check if matrix is symmetric
def symmetry(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True
from itertools import permutations
# Try to find row permutation that makes matrix strictly diagonally dominant
def find_dd_permutation(A, b):
    n = len(A)
    for perm in permutations(range(n)):
        P = [A[perm[i]][:] for i in range(n)]
        ok = True
        for i in range(n):
            diag = abs(P[i][i])
            off = sum(abs(P[i][j]) for j in range(n) if j != i)
            if not (diag > off):   # strict check
                ok = False
                break
        if ok:
            bp = [b[perm[i]] for i in range(n)]
            return perm, P, bp
    return None, A, b

# Write iteration table for Jacobi / Gauss-Seidel
def write_iteration_table(f, title, history):
    f.write(title + "\n")
    n = len(history[0]) - 1
    header = "(k)" + "".join([f"     x{i+1}(k)" for i in range(n)]) + "\n"
    f.write(header)
    for row in history:
        f.write("{:<3d}".format(row[0]))
        for val in row[1:]:
            f.write(" {:10.6f}".format(val))
        f.write("\n")
    f.write("\n")

# Main Program
if __name__ == "__main__":
    with open("assgn5_output.txt", "w") as f:
        #Problem 1
        A1, b1 = read_augmented_matrix("assgn5_agumat01.txt")

        # Cholesky
        if symmetry(A1):
            chol = CholeskySolver(A1, b1)
            x_chol = chol.solve()
            f.write("Problem 1: Cholesky Decomposition Solution\n")
            for i, val in enumerate(x_chol, 1):
                f.write(f"x{i} = {val:.6f}\n")
            f.write("\n")
        else:
            f.write("Problem 1: Matrix not symmetric, cannot use Cholesky\n\n")

        # Gauss-Seidel with iteration table
        gs1 = GaussSeidelSolver(A1, b1)
        x_gs1 = gs1.solve()
        write_iteration_table(f, "Problem 1: Gauss-Seidel Iteration Results", gs1.history)
        f.write("Problem 1: Gauss-Seidel Final Solution\n")
        for i, val in enumerate(x_gs1, 1):
            f.write(f"x{i} = {val:.6f}\n")
        f.write("\n")

        #Problem 2
        A2, b2 = read_augmented_matrix("assgn5_agumat02.txt")

        perm, A2p, b2p = find_dd_permutation(A2, b2)

        # Jacobi with iteration table
        jacobi = JacobiSolver(A2p, b2p)
        x_jacobi = jacobi.solve()
        write_iteration_table(f, "Problem 2: Jacobi Iteration Results", jacobi.history)
        f.write("Problem 2: Jacobi Final Solution\n")
        for i, val in enumerate(x_jacobi, 1):
            f.write(f"x{i} = {val:.6f}\n")
        f.write("\n")

        # Gauss-Seidel with iteration table
        gs2 = GaussSeidelSolver(A2p, b2p)
        x_gs2 = gs2.solve()
        write_iteration_table(f, "Problem 2: Gauss-Seidel Iteration Results", gs2.history)
        f.write("Problem 2: Gauss-Seidel Final Solution\n")
        for i, val in enumerate(x_gs2, 1):
            f.write(f"x{i} = {val:.6f}\n")
        f.write("\n")
