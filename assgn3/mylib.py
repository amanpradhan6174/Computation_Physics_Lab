# mylib.py
# Utility library with Random Generator, Gauss-Jordan Solver, and LU Decomposition (Doolittle)

class LCG:
    """Linear Congruential Generator"""
    def __init__(self, seed=1):
        self.a = 1103515245
        self.c = 12345
        self.m = 32768
        self.state = seed

    def lcg(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state


class GaussJordanSolver:
    """Gauss-Jordan Elimination Solver with step logging"""
    def __init__(self, matrix):
        # Copy matrix so original is not modified
        self.mat = [row[:] for row in matrix]
        self.n = len(matrix)         # number of equations (rows)
        self.log = []

    def print_matrix(self, msg):
        # Save step message + matrix
        self.log.append(msg)
        for row in self.mat:
            line = " ".join(f"{x:10.4f}" for x in row)
            self.log.append(line)
        self.log.append('')

    def solve(self):
        for i in range(self.n):  # go through each pivot column
            # Step 1: partial pivoting
            max_row = i
            for r in range(i + 1, self.n):
                if abs(self.mat[r][i]) > abs(self.mat[max_row][i]):
                    max_row = r
            if max_row != i:
                self.mat[i], self.mat[max_row] = self.mat[max_row], self.mat[i]
                self.print_matrix(f"Swapped row {i+1} with row {max_row+1}")
            else:
                self.print_matrix(f"No swap needed for column {i+1}")

            # Step 2: scale pivot row so pivot = 1
            pivot = self.mat[i][i]
            if pivot == 0:
                raise ValueError(f"Zero pivot at column {i+1}, system may be singular!")
            self.mat[i] = [x / pivot for x in self.mat[i]]
            self.print_matrix(f"Row {i+1} divided by pivot")

            # Step 3: clear column i in all other rows
            for j in range(self.n):
                if j != i:
                    factor = self.mat[j][i]
                    self.mat[j] = [
                        self.mat[j][k] - factor * self.mat[i][k]
                        for k in range(self.n + 1)
                    ]
            self.print_matrix(f"Cleared column {i+1}")

        # Step 4: extract solution (last column)
        solution = [self.mat[i][-1] for i in range(self.n)]
        self.log.append("Final Answer: " + " ".join(f"x{i+1}={solution[i]:.4f}" for i in range(self.n)))
        return solution

    def get_log(self):
        return "\n".join(self.log)


class LUDecomposition:
    
   # LU Decomposition (Doolittle) 
    def __init__(self, A):
        self.A = [row[:] for row in A]  # copy matrix
        self.n = len(A)

    def decompose(self):
        n = self.n
        A = self.A
        for i in range(n):
            # Upper triangular
            for j in range(i, n):
                sum_val = sum(A[i][k] * A[k][j] for k in range(i))
                A[i][j] = A[i][j] - sum_val
            # Lower triangular
            for j in range(i+1, n):
                sum_val = sum(A[j][k] * A[k][i] for k in range(i))
                A[j][i] = (A[j][i] - sum_val) / A[i][i]
        return A
