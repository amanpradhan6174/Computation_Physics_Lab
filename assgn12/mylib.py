#   mylib.py
#   Name - Aman Pradhan 
# Roll Number - 2311020
import math
  

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


    def inverse(A):
        
        #Compute matrix inverse using Gauss-Jordan elimination (pure Python)
        
        n = len(A)
        # Create augmented matrix [A | I]
        aug = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(A)]

        for i in range(n):
            # Pivot: swap row if pivot is zero
            if math.isclose(aug[i][i], 0.0):
                for j in range(i + 1, n):
                    if not math.isclose(aug[j][i], 0.0):
                        aug[i], aug[j] = aug[j], aug[i]
                        break

            # Scale pivot row
            pivot = aug[i][i]
            if math.isclose(pivot, 0.0):
                raise ValueError("Matrix is singular and cannot be inverted!")
            aug[i] = [x / pivot for x in aug[i]]

            # Eliminate column i from other rows
            for j in range(n):
                if j != i:
                    factor = aug[j][i]
                    aug[j] = [aug[j][k] - factor * aug[i][k] for k in range(2 * n)]

        # Extract inverse (right half of augmented matrix)
        inv = [row[n:] for row in aug]
        return inv




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

import math

# ---------------- Fixed-Point Method ---------------- #
class FixedPointMethodMultivar:
    """
    Generalized Fixed-Point Method for n-dimensional nonlinear systems.
    g_funcs: list of functions [g1, g2, ..., gn], each gi(x) takes the vector x as input
    """
    def __init__(self, g_funcs, g_exprs=None, tol=1e-6, max_iter=100):
        self.g_funcs = g_funcs
        self.g_exprs = g_exprs if g_exprs is not None else [f"g{i+1}(x)" for i in range(len(g_funcs))]
        self.n = len(g_funcs)
        self.tol = tol
        self.max_iter = max_iter

    def iterate(self, x0, filename="fixed_point_output.txt", return_history=False):
        x = [float(val) for val in x0]
        n = self.n
        history = []

        with open(filename, "w") as f:
            f.write("Fixed-Point Iteration Method (Generalized)\n")
            f.write("Functions used:\n")
            for i, g_expr in enumerate(self.g_exprs):
                f.write(f"  g{i+1}(x) = {g_expr}\n")
            f.write(f"\nInitial guess: [{', '.join(f'{val:.6f}' for val in x)}]\n\n")

            # Header
            header = "Iter".ljust(8)
            for i in range(n):
                header += f"x{i+1}".rjust(12)
            for i in range(n):
                header += f"x{i+1}'".rjust(12)
            header += "Error".rjust(14) + "\n"
            sep_line = "-"*len(header)
            f.write(sep_line + "\nIteration Table\n" + sep_line + "\n" + header + sep_line + "\n")

            for k in range(1, self.max_iter+1):
                x_new = [self.g_funcs[i](x) for i in range(n)]
                e = max(abs(x_new[i]-x[i]) for i in range(n))
                if return_history:
                    history.append((x_new[:], e))  # simple shallow copy

                row = f"{k:<8d}" + "".join([f"{val:12.6f}" for val in x]) + "".join([f"{val:12.6f}" for val in x_new]) + f"{e:14.6e}\n"
                f.write(row)
                x = x_new
                if e < self.tol:
                    break

            f.write(sep_line + "\n")
            f.write("Final solution: " + " ".join([f"x{i+1}={x[i]:.6f}" for i in range(n)]) + "\n")
            f.write(f"Converged in {k} iterations starting from initial guess [{', '.join(f'{val:.6f}' for val in x0)}]\n")

        return (x, history) if return_history else x


# ---------------- Newton-Raphson Method ---------------- #
class NewtonRaphsonMultivar:
    """
    Newton-Raphson Method for n-dimensional nonlinear systems.
    Self-contained with local Gauss-Jordan inversion.
    """
    def __init__(self, f_funcs, tol=1e-6, max_iter=100, h=1e-6):
        self.f_funcs = f_funcs
        self.n = len(f_funcs)
        self.tol = tol
        self.max_iter = max_iter
        self.h = h

    def jacobian(self, x):
        n = self.n
        J = [[0.0]*n for _ in range(n)]
        for i in range(n):
            f0 = self.f_funcs[i](x)
            for j in range(n):
                x_step = x[:]
                x_step[j] += self.h
                f_step = self.f_funcs[i](x_step)
                J[i][j] = (f_step - f0)/self.h
        return J

    def inverse(self, A):
        n = len(A)
        aug = [row[:] + [float(i==j) for j in range(n)] for i,row in enumerate(A)]
        for i in range(n):
            pivot = aug[i][i]
            if abs(pivot) < 1e-12:
                raise ValueError("Matrix is singular")
            for j in range(2*n):
                aug[i][j] /= pivot
            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2*n):
                        aug[k][j] -= factor*aug[i][j]
        return [row[n:] for row in aug]

    def mat_vec_mult(self, A, v):
        return [sum(A[i][j]*v[j] for j in range(self.n)) for i in range(self.n)]

    def vec_sub(self, v1, v2):
        return [v1[i]-v2[i] for i in range(len(v1))]

    def norm_inf(self, v):
        return max(abs(val) for val in v)

    def iterate(self, x0, filename="output.txt", return_history=False, append=False):
        """
        Solve the system using Newton-Raphson.
        append=True -> append to existing file; False -> overwrite
        """
        x = [float(val) for val in x0]
        n = self.n
        history = []

        mode = "a" if append else "w"
        with open(filename, mode) as f:
            f.write("\nNewton-Raphson Method (Multivariable)\n")
            header = "Iter".ljust(8)
            header += "".join([f"x{i+1}".rjust(12) for i in range(n)])
            header += "".join([f"f{i+1}".rjust(14) for i in range(n)])
            header += "Error".rjust(14) + "\n"
            sep_line = "-"*len(header)
            f.write(sep_line + "\nIteration Table\n" + sep_line + "\n" + header + sep_line + "\n")

            for k in range(1, self.max_iter+1):
                F = [self.f_funcs[i](x) for i in range(n)]
                J = self.jacobian(x)
                J_inv = self.inverse(J)
                JF = self.mat_vec_mult(J_inv, F)
                x_new = self.vec_sub(x, JF)
                e = self.norm_inf([x_new[i]-x[i] for i in range(n)])
                if return_history:
                    history.append((x_new[:], F[:], e))

                row = f"{k:<8d}" + "".join([f"{val:12.6f}" for val in x]) \
                      + "".join([f"{val:14.6e}" for val in F]) + f"{e:14.6e}\n"
                f.write(row)

                x = x_new
                if e < self.tol:
                    break

            f.write(sep_line + "\n")
            f.write("Final solution: " + " ".join([f"x{i+1}={x[i]:.6f}" for i in range(n)]) + "\n")
            f.write(f"Converged in {k} iterations starting from initial guess [{', '.join(f'{val:.6f}' for val in x0)}]\n")

        return (x, history) if return_history else x


import math

class LaguerreDeflationSolver:
    def __init__(self, coeffs, tol=1e-6, max_iter=100, filename="output.txt"):
        # Save polynomial coefficients and settings
        self.coeffs = coeffs[:]
        self.tol = tol
        self.max_iter = max_iter
        self.filename = filename

    # ---------------- Polynomial Utilities ---------------- #
    def poly_eval(self, coeffs, x):
        # Evaluate polynomial at x (Horner’s method)
        result = coeffs[0]
        for c in coeffs[1:]:
            result = result * x + c
        return result

    def poly_derivative(self, coeffs):
        # Return derivative coefficients
        n = len(coeffs) - 1
        return [coeffs[i] * (n - i) for i in range(len(coeffs)-1)]

    def poly_to_string(self, coeffs):
        # Convert polynomial coefficients to a readable string
        n = len(coeffs) - 1
        terms = []
        for i, c in enumerate(coeffs):
            power = n - i
            if abs(c) < 1e-12:
                continue
            if power == 0:
                terms.append(f"{c:.4f}")
            elif power == 1:
                terms.append(f"{c:.4f}x")
            else:
                terms.append(f"{c:.4f}x^{power}")
        return " + ".join(terms).replace("+ -", "- ")

    # ---------------- Laguerre Method ---------------- #
    def laguerre(self, coeffs, x0):
        # Find one root using Laguerre’s method
        n = len(coeffs) - 1
        x = x0

        # Write table header
        with open(self.filename, "a") as f:
            f.write("\nIteration Table:\n")
            f.write("{:<6} {:<15} {:<15} {:<15} {:<15} {:<15}\n".format(
                "Iter", "x_k", "P(x_k)", "G(x)", "H(x)", "Dx"
            ))
            f.write("-" * 85 + "\n")

        for k in range(self.max_iter):
            # Evaluate polynomial and derivatives
            p = self.poly_eval(coeffs, x)
            p1 = self.poly_eval(self.poly_derivative(coeffs), x)
            p2 = self.poly_eval(self.poly_derivative(self.poly_derivative(coeffs)), x)

            if abs(p) < 1e-14:
                break

            # G and H terms
            G = p1 / p
            H = G * G - (p2 / p)

            # Choose denominator
            denom1 = G + math.sqrt((n - 1) * (n * H - G * G))
            denom2 = G - math.sqrt((n - 1) * (n * H - G * G))
            if abs(denom1) > abs(denom2):
                a = n / denom1
            else:
                a = n / denom2

            # Next guess
            x_new = x - a

            # Save row in file
            with open(self.filename, "a") as f:
                f.write("{:<6d} {:<15.8f} {:<15.8e} {:<15.8f} {:<15.8f} {:<15.8f}\n".format(
                    k, x, p, G, H, -a
                ))

            # Stop if converged
            if abs(p) < self.tol or abs(x_new - x) < self.tol:
                return x_new

            x = x_new

        return x

    # ---------------- Deflation ---------------- #
    def deflate(self, coeffs, root):
        # Divide polynomial by (x - root)
        n = len(coeffs) - 1
        new_coeffs = [coeffs[0]]
        for i in range(1, n+1):
            new_coeffs.append(coeffs[i] + root * new_coeffs[-1])
        return new_coeffs[:-1]

    # ---------------- Main Solver ---------------- #
    def solve(self, poly_name="P(x)"):
        roots = []
        coeffs = self.coeffs[:]

        # Start new file
        with open(self.filename, "a") as f:
            f.write("Laguerre Method Solution\n")
            f.write("="*85 + "\n")
            f.write(f"Given Polynomial: {poly_name}\n")
            f.write(f"{self.poly_to_string(coeffs)}\n")
            f.write("="*85 + "\n")

        # Solve until polynomial reduces to quadratic/linear
        while len(coeffs) > 2:
            root = self.laguerre(coeffs, 0.0)
            root = round(root, 8)
            roots.append(root)

            # Deflate polynomial
            coeffs = self.deflate(coeffs, root)

            with open(self.filename, "a") as f:
                f.write(f"\nFound root: {root}\n")
                f.write("="*85 + "\n")

        # Handle last linear root
        if len(coeffs) == 2:
            root = -coeffs[1] / coeffs[0]
            roots.append(round(root, 8))
            with open(self.filename, "a") as f:
                f.write(f"\nFinal root: {root}\n")

        # Write all roots
        with open(self.filename, "a") as f:
            f.write("\nAll roots:\n")
            f.write(str(roots) + "\n")
            f.write("="*85 + "\n")

        return roots




# MIDPOINT INTEGRATION CLASS
class MidpointIntegration:
    def __init__(self, func, func_dd, a, b, exact_value, N_values, name):
        """
        Initialize the Midpoint Integration solver.
        func     : function to integrate
        func_dd  : second derivative of function (for error bound)
        a, b     : integration limits
        exact_value : analytical (true) value of the integral
        N_values : list of subdivisions to test (e.g. [4, 8, 15, 20])
        name     : descriptive name of the problem
        """
        self.f = func
        self.f_dd = func_dd
        self.a = a
        self.b = b
        self.exact = exact_value
        self.N_values = N_values
        self.name = name

  
    # Find maximum |f''(x)| on [a, b] using 1000 samples
   
    def find_max_second_derivative(self):
        steps = 1000
        dx = (self.b - self.a) / steps
        f2max = 0.0
        for i in range(steps + 1):
            x = self.a + i * dx
            val = abs(self.f_dd(x))
            if val > f2max:
                f2max = val
        return f2max

    
    # Error bound for Midpoint rule
    # Formula: E_mid = ((b - a)^3 / (24*N^2)) * max|f''(x)|
    
    def error_bound(self, N, f2max):
        return ((self.b - self.a) ** 3) / (24 * (N ** 2)) * f2max

    # Midpoint Integration core computation
   
    def integrate(self, N):
        a, b = self.a, self.b
        h = (b - a) / N       # step size
        total = 0.0           # running sum of f(x_mid)
        steps = []            # store process for process.txt

        for i in range(N):
            x_mid = a + (i + 0.5) * h
            fx = self.f(x_mid)
            total += fx
            steps.append((i + 1, x_mid, fx, total * h))  # record current partial sum

        return h * total, steps  # return integral value and process list

    
    # Perform full computation, write results and process
    
    def compute_and_write(self, output_file, process_file):
        f2max = self.find_max_second_derivative()

        # open files for appending text results
        with open(output_file, "a", encoding="utf-8") as f_out, open(process_file, "a", encoding="utf-8") as f_proc:
            # --- Write final summary to output.txt ---
            f_out.write(f"\n{'='*70}\n")
            f_out.write(f"Midpoint Integration Results for {self.name}\n")
            f_out.write(f"{'='*70}\n")
            f_out.write("N\tApproximation\tExact\t\tError\t\tError Bound\n")

            # Loop over each N and compute integral
            for N in self.N_values:
                approx, steps = self.integrate(N)
                error = abs(approx - self.exact)
                E_mid = self.error_bound(N, f2max)

                # Write one summary line per N
                f_out.write("%d\t%.8f\t%.8f\t%.8e\t%.8e\n" % (N, approx, self.exact, error, E_mid))

                # --- Write detailed iteration steps to process.txt ---
                f_proc.write(f"\n--- Midpoint Integration ({self.name}) for N={N} ---\n")
                f_proc.write("Step\tX_mid\t\tf(X_mid)\tPartial Sum\n")
                for s in steps:
                    f_proc.write("%d\t%.8f\t%.8f\t%.8f\n" % s)
                f_proc.write("\n")



# TRAPEZOIDAL INTEGRATION CLASS

class TrapezoidalIntegration:
    def __init__(self, func, func_dd, a, b, exact_value, N_values, name):
        """
        Initialize the Trapezoidal Integration solver.
        Parameters are the same as in MidpointIntegration.
        """
        self.f = func
        self.f_dd = func_dd
        self.a = a
        self.b = b
        self.exact = exact_value
        self.N_values = N_values
        self.name = name

    # Find maximum |f''(x)| on [a, b]

    def find_max_second_derivative(self):
        steps = 1000
        dx = (self.b - self.a) / steps
        f2max = 0.0
        for i in range(steps + 1):
            x = self.a + i * dx
            val = abs(self.f_dd(x))
            if val > f2max:
                f2max = val
        return f2max

    # Error bound for Trapezoidal rule
    # Formula: E_trap = ((b - a)^3 / (12*N^2)) * max|f''(x)|
    
    def error_bound(self, N, f2max):
        return ((self.b - self.a) ** 3) / (12 * (N ** 2)) * f2max

   
    # Trapezoidal Integration computation
  
    def integrate(self, N):
        a, b = self.a, self.b
        h = (b - a) / N
        total = 0.5 * (self.f(a) + self.f(b))  # first and last terms weighted 1/2
        steps = [(0, a, self.f(a), total * h)]  # store steps for process file

        # Sum up the middle points
        for i in range(1, N):
            x = a + i * h
            fx = self.f(x)
            total += fx
            steps.append((i, x, fx, total * h))

        steps.append((N, b, self.f(b), total * h))
        return h * total, steps

    # Perform full computation, write results and process
  
    def compute_and_write(self, output_file, process_file):
        f2max = self.find_max_second_derivative()

        with open(output_file, "a", encoding="utf-8") as f_out, open(process_file, "a", encoding="utf-8") as f_proc:
            # --- Write final summary to output.txt ---
            f_out.write(f"\n{'='*70}\n")
            f_out.write(f"Trapezoidal Integration Results for {self.name}\n")
            f_out.write(f"{'='*70}\n")
            f_out.write("N\tApproximation\tExact\t\tError\t\tError Bound\n")

            for N in self.N_values:
                approx, steps = self.integrate(N)
                error = abs(approx - self.exact)
                E_trap = self.error_bound(N, f2max)

                # Write one summary line per N
                f_out.write("%d\t%.8f\t%.8f\t%.8e\t%.8e\n" % (N, approx, self.exact, error, E_trap))

                # --- Write detailed iteration steps ---
                f_proc.write(f"\n--- Trapezoidal Integration ({self.name}) for N={N} ---\n")
                f_proc.write("Step\tX_i\t\tf(X_i)\tPartial Sum\n")
                for s in steps:
                    f_proc.write("%d\t%.8f\t%.8f\t%.8f\n" % s)
                f_proc.write("\n")



# ============================================================
# Midpoint Rule with N finding
# ============================================================

class MidpointNfindingIntegration:
    def __init__(self, func, func_dd, a, b, exact_value, name):
        self.f = func
        self.f_dd = func_dd
        self.a = a
        self.b = b
        self.exact = exact_value
        self.name = name

    # Find N for given tolerance using error bound
    def find_required_N(self, tol):
        steps = 2000
        dx = (self.b - self.a) / steps
        f2max = 0.0
        for i in range(steps + 1):
            x = self.a + i * dx
            val = abs(self.f_dd(x))
            if val > f2max:
                f2max = val
        numerator = ((self.b - self.a) ** 3) * f2max
        denom = 24.0 * tol
        N_int = int(math.ceil(math.sqrt(numerator / denom)))
        return max(1, N_int), f2max

    def error_bound(self, N, f2max):
        return ((self.b - self.a) ** 3) / (24.0 * (N ** 2)) * f2max

    # Perform integration
    def integrate(self, N):
        a, b = self.a, self.b
        h = (b - a) / N
        total, steps = 0.0, []
        for i in range(N):
            x_mid = a + (i + 0.5) * h
            fx = self.f(x_mid)
            total += fx
            steps.append((i + 1, x_mid, fx, total * h))
        return h * total, steps

    def compute_and_write(self, output_file, process_file, tol):
        N, f2max = self.find_required_N(tol)
        approx, steps = self.integrate(N)
        error = abs(approx - self.exact)
        E_mid = self.error_bound(N, f2max)

        with open(output_file, "w", encoding="utf-8") as f_out, open(process_file, "w", encoding="utf-8") as f_proc:
            f_out.write(f"\n{'='*70}\nMidpoint Integration ({self.name})\n{'='*70}\n")
            f_out.write(f"N={N}, max|f''(x)|≈{f2max:.6f}, tol={tol:.1e}\n")
            f_out.write("N\tApproximation\tExact\t\tError\t\tError Bound\n")
            f_out.write("%d\t%.8f\t%.8f\t%.8e\t%.8e\n" % (N, approx, self.exact, error, E_mid))

            f_proc.write(f"\n--- Midpoint Process for {self.name}, N={N} ---\n")
            f_proc.write("Step\tX_mid\t\tf(X_mid)\tPartial Integral\n")
            for s in steps:
                f_proc.write("%d\t%.8f\t%.8f\t%.8f\n" % s)
            f_proc.write("\n")


# ============================================================
# Simpson's 1/3 Rule with N finding
# ============================================================

class SimpsonIntegration:
    def __init__(self, func, func_4th, a, b, exact_value, name):
        self.f = func
        self.f_4 = func_4th
        self.a = a
        self.b = b
        self.exact = exact_value
        self.name = name

    # Find N for given tolerance using Simpson error bound
    def find_required_N(self, tol):
        steps = 2000
        dx = (self.b - self.a) / steps
        f4max = 0.0
        for i in range(steps + 1):
            x = self.a + i * dx
            val = abs(self.f_4(x))
            if val > f4max:
                f4max = val
        numerator = ((self.b - self.a) ** 5) * f4max
        denom = 180.0 * tol
        N_int = int(math.ceil((numerator / denom) ** 0.25))
        if N_int % 2 == 1:
            N_int += 1
        return max(2, N_int), f4max

    def error_bound(self, N, f4max):
        return ((self.b - self.a) ** 5) / (180.0 * (N ** 4)) * f4max

    # Perform integration
    def integrate(self, N):
        if N % 2 == 1:
            N += 1
        a, b = self.a, self.b
        h = (b - a) / N
        total = self.f(a) + self.f(b)
        steps = []
        for i in range(1, N):
            x = a + i * h
            coef = 4 if (i % 2 == 1) else 2
            fx = self.f(x)
            total += coef * fx
            steps.append((i, x, coef, fx, (h/3.0)*total))
        return (h / 3.0) * total, steps

    def compute_and_write(self, output_file, process_file, tol):
        N, f4max = self.find_required_N(tol)
        approx, steps = self.integrate(N)
        error = abs(approx - self.exact)
        E_simp = self.error_bound(N, f4max)

        with open(output_file, "w", encoding="utf-8") as f_out, open(process_file, "w", encoding="utf-8") as f_proc:
            f_out.write(f"\n{'='*70}\nSimpson Integration ({self.name})\n{'='*70}\n")
            f_out.write(f"N={N}, max|f⁽⁴⁾(x)|≈{f4max:.6f}, tol={tol:.1e}\n")
            f_out.write("N\tApproximation\tExact\t\tError\t\tError Bound\n")
            f_out.write("%d\t%.8f\t%.8f\t%.8e\t%.8e\n" % (N, approx, self.exact, error, E_simp))

            f_proc.write(f"\n--- Simpson Process for {self.name}, N={N} ---\n")
            f_proc.write("i\tX_i\t\tCoef\tf(X_i)\tPartial Simpson\n")
            for s in steps:
                i, x, coef, fx, partial = s
                f_proc.write("%d\t%.8f\t%d\t%.8f\t%.8f\n" % (i, x, coef, fx, partial))
            f_proc.write("\n")

import math
import matplotlib.pyplot as plt


#  Custom Linear Congruential Generator (LCG)

class LCG:
    def __init__(self, seed=1):
        # Use large modulus to avoid repeating small cycles
        self.a = 1103515245
        self.c = 12345
        self.m = 2**31 - 1   # much larger than 32768
        self.state = seed

    def lcg(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def uniform(self):
        """Return random float in [0, 1)."""
        return self.lcg() / float(self.m)

#  Monte Carlo Integration 


class MonteCarloIntegration:
    def __init__(self, func, a, b, exact_value=None, lcg_seed=1, name="Monte Carlo"):
        self.f = func
        self.a = a
        self.b = b
        self.exact = exact_value
        self.lcg_seed = lcg_seed
        self.name = name

    def estimate(self, N):
        """Compute Monte Carlo estimate with independent samples."""
        rng = LCG(seed=self.lcg_seed)
        fx_vals = []

        for _ in range(N):
            u = rng.uniform()
            x = self.a + (self.b - self.a) * u
            fx_vals.append(self.f(x))

        mean_f = sum(fx_vals) / N
        mean_f2 = sum(v * v for v in fx_vals) / N
        sigma_f = math.sqrt(max(0.0, mean_f2 - mean_f ** 2))

        F_N = (self.b - self.a) * mean_f
        sigma_F = (self.b - self.a) * sigma_f / math.sqrt(N)
        return F_N, sigma_F

    def run_until_tol(
        self,
        tol_abs=1e-3,
        N_start=100,
        N_step=100,
        N_max=500000,
        output_file="output2.txt",
        process_file="process2.txt",
        plot_file="mc_plot.png"
    ):
        Ns, estimates, errors = [], [], []

        with open(process_file, "w", encoding="utf-8") as pf:
            pf.write(f"{'='*70}\nMonte Carlo Process for {self.name}\n{'='*70}\n")
            pf.write("N\tEstimate F_N\t|F_N - Exact|\n")

            stop_reason = "Max samples reached"
            N = N_start

            while N <= N_max:
                FN, sigma = self.estimate(N)
                err = abs(FN - self.exact) if self.exact is not None else float("nan")

                Ns.append(N)
                estimates.append(FN)
                errors.append(err)

                pf.write(f"{N}\t{FN:.8f}\t{err:.5e}\n")

                if N % (10 * N_step) == 0:
                    print(f"N = {N:6d} → F_N = {FN:.6f}, |error| = {err:.5e}")

                # Stop when target accuracy reached
                if self.exact is not None and err <= tol_abs:
                    stop_reason = f"|F_N - exact| ≤ {tol_abs}"
                    break

                N += N_step

        # Write summary
        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.write(f"{'='*70}\nMonte Carlo Integration Summary: {self.name}\n{'='*70}\n")
            f_out.write(f"Exact value\t: {self.exact:.8f}\n")
            f_out.write(f"Final estimate\t: {FN:.8f}\n")
            f_out.write(f"Samples used (N)\t: {N}\n")
            f_out.write(f"Absolute error\t: {err:.3e}\n")
            f_out.write(f"Stop reason\t: {stop_reason}\n")

        # Plot convergence
        plt.figure(figsize=(7, 4))
        plt.plot(Ns, estimates, label="Estimated F_N", linewidth=1.5)
        if self.exact is not None:
            plt.axhline(self.exact, color="r", linestyle="--", label="Exact value")
        plt.xlabel("Number of Samples (N)")
        plt.ylabel("Estimated Integral F_N")
        plt.title(f"Monte Carlo Convergence: {self.name}")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150)
        plt.close()

        return FN, N, err
    


# Assignment 12 — Numerical Integration
#   - SimpsonWithoutNFinding (manual N)
#   - GaussianQuadrature (Legendre roots method)


import math


# Gaussian Quadrature Class

class GaussianQuadrature:
    def __init__(self, func, a, b, n, exact_value, name):
        self.f = func
        self.a = a
        self.b = b
        self.n = n
        self.exact = exact_value
        self.name = name

        # Roots (x) and Weights (w) for Legendre Polynomials (n = 2–16)
        self.data = {
            2:  ([0.577350269189626], [1.000000000000000]),
            3:  ([0.000000000000000, 0.774596669241483],
                 [0.888888888888889, 0.555555555555556]),
            4:  ([0.339981043584856, 0.861136311594053],
                 [0.652145154862546, 0.347854845137454]),
            5:  ([0.000000000000000, 0.538469310105683, 0.906179845938664],
                 [0.568888888888889, 0.478628670499366, 0.236926885056189]),
            6:  ([0.238619186083197, 0.661209386466265, 0.932469514203152],
                 [0.467913934572691, 0.360761573048139, 0.171324492379170]),
            7:  ([0.000000000000000, 0.405845151377397, 0.741531185599394, 0.949107912342759],
                 [0.417959183673469, 0.381830050505119, 0.279705391489277, 0.129484966168870]),
            8:  ([0.183434642495650, 0.525532409916329, 0.796666477413627, 0.960289856497536],
                 [0.362683783378362, 0.313706645877887, 0.222381034453374, 0.101228536290376]),
            9:  ([0.000000000000000, 0.324253423403809, 0.613371432700590, 0.836031107326636, 0.968160239507626],
                 [0.330239355001260, 0.312347077040003, 0.260610696402935,
                  0.180648160694857, 0.081274388361574]),
            10: ([0.148874338981631, 0.433395394129247, 0.679409568299024,
                  0.865063366688985, 0.973906528517172],
                 [0.295524224714753, 0.269266719309996, 0.219086362515982,
                  0.149451349150581, 0.066671344308688]),
            11: ([0.000000000000000, 0.269543155952345, 0.519096129110681, 0.730152005574049,
                   0.887062599768095, 0.978228658146057],
                 [0.272925086777901, 0.262804544510247, 0.233193764591990,
                  0.186290210927734, 0.125580369464905, 0.055668567116174]),
            12: ([0.125233408511469, 0.367831498918180, 0.587317954286617,
                   0.769902674194305, 0.904117256370475, 0.981560634246719],
                 [0.249147045813403, 0.233492536538355, 0.203167426723066,
                  0.160078328543346, 0.106939325995318, 0.047175336386512]),
            13: ([0.000000000000000, 0.230458315955135, 0.448492751036447,
                   0.642349339440340, 0.801578090733310, 0.917598399222978,
                   0.984183054718588],
                 [0.232551553230874, 0.226283180262897, 0.207816047536889,
                  0.178145980761946, 0.138873510219787, 0.092121499837728,
                  0.040484004765316]),
            14: ([0.108054948707344, 0.319112368927890, 0.515248636358154,
                   0.687292904811685, 0.827201315069765, 0.928434883663574,
                   0.986283808696812],
                 [0.215263853463158, 0.205198463721290, 0.185538397477938,
                  0.157203167158194, 0.121518570687903, 0.080158087159760,
                  0.035119460331752]),
            15: ([0.000000000000000, 0.201194093997435, 0.394151347077563,
                   0.570972172608539, 0.724417731360170, 0.848206583410427,
                   0.937273392400706, 0.987992518020485],
                 [0.202578241925561, 0.198431485327111, 0.186161000015562,
                  0.166269205816994, 0.139570677926154, 0.107159220467172,
                  0.070366047488108, 0.030753241996117]),
            16: ([0.095012509837637, 0.281603550779259, 0.458016777657227,
                   0.617876244402644, 0.755404408355003, 0.865631202387832,
                   0.944575023073233, 0.989400934991650],
                 [0.189450610455069, 0.182603415044924, 0.169156519395003,
                  0.149595988816577, 0.124628971255534, 0.095158511682493,
                  0.062253523938648, 0.027152459411754])
        }

    # Perform Gaussian Quadrature integration
    def integrate(self):
        a, b = self.a, self.b
        roots, weights = self.data[self.n]
        total = 0.0
        # Symmetric evaluation over ±xi
        for i in range(len(roots)):
            xi = roots[i]
            wi = weights[i]
            total += wi * (self.f((b - a) / 2 * xi + (b + a) / 2)
                           + self.f((b - a) / 2 * (-xi) + (b + a) / 2))
        result = (b - a) / 2 * total
        return result

    # Write output and process details
    def compute_and_write(self, output_file, process_file):
        result = self.integrate()
        error = abs(result - self.exact)

        with open(output_file, "a") as f_out, open(process_file, "a") as f_proc:
            f_out.write("\n" + "="*70 + f"\nGaussian Quadrature ({self.name})\n" + "="*70 + "\n")
            f_out.write(f"n = {self.n}\nApproximation = {result:.9f}\nExact = {self.exact:.9f}\nError = {error:.2e}\n")

            f_proc.write(f"\n--- Gaussian Quadrature Steps ({self.name}) ---\n")
            f_proc.write("i\tRoot\tWeight\n")
            for i in range(len(self.data[self.n][0])):
                f_proc.write(f"{i+1}\t{self.data[self.n][0][i]:.9f}\t{self.data[self.n][1][i]:.9f}\n")

        return result, error



# Simpson's Rule (Manual N specification)

class SimpsonWithoutNFinding:
    def __init__(self, func, a, b, exact_value, name):
        self.f = func
        self.a = a
        self.b = b
        self.exact = exact_value
        self.name = name

    # Perform Simpson integration for given N
    def integrate(self, N):
        if N % 2 == 1:
            N += 1
        a, b = self.a, self.b
        h = (b - a) / N
        total = self.f(a) + self.f(b)
        for i in range(1, N):
            x = a + i * h
            coef = 4 if (i % 2 == 1) else 2
            total += coef * self.f(x)
        result = (h / 3.0) * total
        return result

    # Compute, compare with exact, and write outputs
    def compute_and_write(self, output_file, process_file, N):
        result = self.integrate(N)
        error = abs(result - self.exact)

        with open(output_file, "a") as f_out, open(process_file, "a") as f_proc:
            f_out.write("\n" + "="*70 + f"\nSimpson (Manual N) Integration ({self.name})\n" + "="*70 + "\n")
            f_out.write(f"N = {N}\nApproximation = {result:.9f}\nExact = {self.exact:.9f}\nError = {error:.2e}\n")

            f_proc.write(f"\n--- Simpson Steps ({self.name}), N={N} ---\n")
            f_proc.write("i\tX_i\tCoef\tf(X_i)\n")
            a, b = self.a, self.b
            h = (b - a) / N
            for i in range(N + 1):
                x = a + i * h
                coef = 1 if i == 0 or i == N else (4 if i % 2 == 1 else 2)
                f_proc.write(f"{i}\t{x:.8f}\t{coef}\t{self.f(x):.8f}\n")

        return result, error
