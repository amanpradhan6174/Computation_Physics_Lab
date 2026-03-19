# mylib.py
# Utility library with Random Generator, Gauss-Jordan Solver, and LU Decomposition (Doolittle)

class LCG:
    # Linear Congruential Generator
    def __init__(self, seed=1):
        self.a = 1103515245
        self.c = 12345
        self.m = 32768
        self.state = seed

    def lcg(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state


class GaussJordanSolver:
    # Gauss-Jordan Elimination Solver with step logging
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

import math

class CholeskySolver:
    # works if A is symmetric and positive definite.

    def __init__(self, A, b):
        self.A = [row[:] for row in A]   # copy matrix A
        self.b = b[:]                    # copy vector b
        self.n = len(A)                  # number of rows
        self.L = [[0.0] * self.n for _ in range(self.n)]  # empty lower matrix

    def decompose(self):
        # Build lower triangular matrix L
        for i in range(self.n):
            for j in range(i + 1):
                sum_val = 0.0
                for k in range(j):
                    sum_val += self.L[i][k] * self.L[j][k]

                if i == j:
                    self.L[i][j] = math.sqrt(self.A[i][i] - sum_val)  # diagonal
                else:
                    self.L[i][j] = (self.A[i][j] - sum_val) / self.L[j][j]  # off-diagonal
        return self.L

    def forward(self, L, b):
        # Solve L * y = b
        n = len(L)
        y = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(i):
                s += L[i][j] * y[j]
            y[i] = (b[i] - s) / L[i][i]
        return y

    def backward(self, U, y):
        # Solve U * x = y  (where U = L^T)
        n = len(U)
        x = [0.0] * n
        for i in range(n-1, -1, -1):
            s = 0.0
            for j in range(i+1, n):
                s += U[i][j] * x[j]
            x[i] = (y[i] - s) / U[i][i]
        return x

    def solve(self):
        # Full process: A = L * L^T -> solve
        L = self.decompose()
        y = self.forward(L, self.b)       # forward substitution
        x = self.backward([list(row) for row in zip(*L)], y)  # pass L^T into backward
        return x
class JacobiSolver:

   # Solve Ax = b using Jacobi Iterative Method. Works for any size system.

    def __init__(self, A, b, tol=1e-6, max_iter=1000):
        self.A = [row[:] for row in A]   # copy of A
        self.b = b[:]                    # copy of b
        self.n = len(A)                  # number of equations
        self.tol = tol                   # stopping tolerance
        self.max_iter = max_iter         # max number of iterations
        self.history = []                # to store iteration results

    def solve(self):
        # Start with initial guess = zeros
        x_old = [0.0 for _ in range(self.n)]
        self.history.append([0] + x_old[:])  # save first guess

        for iteration in range(1, self.max_iter + 1):
            x_new = [0.0 for _ in range(self.n)]

            # Jacobi formula
            for i in range(self.n):
                s = 0.0
                for j in range(self.n):
                    if j != i:
                        s += self.A[i][j] * x_old[j]
                x_new[i] = (self.b[i] - s) / self.A[i][i]

            # save result of this iteration
            self.history.append([iteration] + x_new[:])

            # check convergence
            error = max(abs(x_new[i] - x_old[i]) for i in range(self.n))
            if error < self.tol:
                return x_new

            x_old = x_new[:]

        return x_new  # return last result if not converged

    def verify(self, x):
        # Check if A*x ≈ b
        Ax = [0.0] * self.n
        for i in range(self.n):
            s = 0.0
            for j in range(self.n):
                s += self.A[i][j] * x[j]
            Ax[i] = s
        ok = all(abs(Ax[i] - self.b[i]) < 1e-6 for i in range(self.n))
        return ok, Ax
class GaussSeidelSolver:
    # Solve Ax = b using Gauss-Seidel Iterative Method (with iteration history)

    def __init__(self, A, b, tol=1e-6, max_iter=1000):
        self.A = [row[:] for row in A]   # copy matrix A
        self.b = b[:]                    # copy vector b
        self.n = len(A)                  # number of equations
        self.tol = tol                   # tolerance for stopping
        self.max_iter = max_iter         # maximum number of iterations
        self.history = []                 # store iteration history

    def solve(self):
        # Start with x = zeros
        x = [0.0 for _ in range(self.n)]
        self.history.append([0] + x[:])   # save iteration 0

        for k in range(1, self.max_iter + 1):
            x_old = x[:]
            for i in range(self.n):
                s1 = sum(self.A[i][j] * x[j] for j in range(i))       # new values
                s2 = sum(self.A[i][j] * x_old[j] for j in range(i+1, self.n)) # old values
                x[i] = (self.b[i] - s1 - s2) / self.A[i][i]

            # save current iteration
            self.history.append([k] + x[:])

            # check convergence
            error = max(abs(x[i] - x_old[i]) for i in range(self.n))
            if error < self.tol:
                return x
        return x


# mylib.py
import math


# Bisection Method Class

class BisectionMethod:
    def __init__(self, func, a, b, tol=1e-6, max_iter=1000):
        # function to solve, interval [a,b], tolerance, max iterations
        self.func = func
        self.a = a
        self.b = b
        self.tol = tol
        self.max_iter = max_iter
        self.iterations = []   # store each iteration for later printing

    def solve(self):
        a, b = self.a, self.b
        fa, fb = self.func(a), self.func(b)

        # quick check: if endpoint itself is root
        if abs(fa) < self.tol: 
            self.iterations.append((0,a,b,a,fa,fb,fa))
            return a
        if abs(fb) < self.tol: 
            self.iterations.append((0,a,b,b,fa,fb,fb))
            return b

        # root must be bracketed (sign change)
        if fa * fb > 0:
            raise ValueError("Bisection: f(a) and f(b) must have opposite signs.")

        # main loop
        for i in range(1, self.max_iter + 1):
            c = (a + b) / 2.0     # midpoint
            fc = self.func(c)
            self.iterations.append((i, a, b, c, fa, fb, fc))

            # stop if solution close enough
            if abs(fc) < self.tol or (b - a) / 2.0 < self.tol:
                return c

            # update interval depending on sign
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc

        # fail-safe
        raise ValueError("Bisection did not converge within max_iter.")


# Regula Falsi (False Position) Method Class

class RegulaFalsiMethod:
    def __init__(self, func, a, b, tol=1e-6, max_iter=1000):
        # function to solve, interval [a,b], tolerance, max iterations
        self.func = func
        self.a = a
        self.b = b
        self.tol = tol
        self.max_iter = max_iter
        self.iterations = []   # store each iteration for later printing

    def solve(self):
        a, b = self.a, self.b
        fa, fb = self.func(a), self.func(b)

        # quick check: if endpoint itself is root
        if abs(fa) < self.tol:
            self.iterations.append((0,a,b,a,fa,fb,fa))
            return a
        if abs(fb) < self.tol:
            self.iterations.append((0,a,b,b,fa,fb,fb))
            return b

        # root must be bracketed (sign change)
        if fa * fb > 0:
            raise ValueError("Regula Falsi: f(a) and f(b) must have opposite signs.")

        # main loop
        for i in range(1, self.max_iter + 1):
            # formula for "false-position" (linear interpolation)
            c = b - (b - a) * fb / (fb - fa)
            fc = self.func(c)
            self.iterations.append((i, a, b, c, fa, fb, fc))

            # stop if solution close enough
            if abs(fc) < self.tol or abs(b - a) < self.tol:
                return c

            # update interval depending on sign
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc

        # fail-safe
        raise ValueError("Regula Falsi did not converge within max_iter.")


class NewtonRaphsonMethod:
    def __init__(self, f, f_prime, tol=1e-6, max_iter=100):
        self.f = f
        self.f_prime = f_prime
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, x0):
        iterations = []
        x = x0

        for i in range(1, self.max_iter + 1):
            fx = self.f(x)
            fpx = self.f_prime(x)
            if fpx == 0:
                raise ZeroDivisionError("Derivative became zero.")

            x_new = x - fx / fpx
            iterations.append((i, x, fx, fpx, x_new))

            if abs(x_new - x) < self.tol:
                return x_new, iterations

            x = x_new

        raise ValueError("Newton-Raphson did not converge.")
    
class FixedPointMethod:
    def __init__(self, g, tol=1e-6, max_iter=100):
        # g(x) is the rearranged function such that x = g(x)
        self.g = g
        self.tol = tol
        self.max_iter = max_iter
        self.iterations = []   # store each iteration (i, xn, g(xn), diff)

    def solve(self, x0, filename="output1.txt"):
        x = x0
        with open(filename, "a") as f:  # append mode
            f.write("\n=== FIXED POINT METHOD ITERATIONS ===\n")
            f.write(f"{'Iter':<6}{'xn':<15}{'g(xn)':<15}{'xn+1 - xn':<15}\n")
        
            for i in range(1, self.max_iter + 1):
                gx = self.g(x)                # g(xn)
                diff = gx - x                 # xn+1 - xn
                self.iterations.append((i, x, gx, diff))

                # write iteration row to file
                f.write(f"{i:<6}{x:<15.8f}{gx:<15.8f}{diff:<15.8f}\n")

                if abs(diff) < self.tol:
                    f.write(f"\nConverged Root = {gx:.6f}\n")
                    return gx  # converged root

                x = gx

        raise ValueError("Fixed Point method did not converge.")
