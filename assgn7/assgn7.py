# assgnment - 7
# Name - Aman Pradhan
# Roll Number - 2311020
import math
from mylib import BisectionMethod, RegulaFalsiMethod, NewtonRaphsonMethod

# Define the function
def f(x):
    return 3*x + math.sin(x) - math.exp(x)

# Derivative for Newton-Raphson
def f_prime(x):
    return 3 + math.cos(x) - math.exp(x)

# Interval for Bisection/Regula Falsi
a, b = -1.5, 1.5

# Initial guess for Newton-Raphson
x0 = 0.5

# Create solver objects
bis_solver = BisectionMethod(f, a, b, tol=1e-6, max_iter=100)
rf_solver = RegulaFalsiMethod(f, a, b, tol=1e-6, max_iter=100)
nr_solver = NewtonRaphsonMethod(f, f_prime, tol=1e-6, max_iter=100)

# Run solvers
root_bis = bis_solver.solve()
root_rf = rf_solver.solve()
root_nr, iter_nr = nr_solver.solve(x0)

# Save results to output1.txt
with open("output1.txt", "w") as fout:
    # Bisection
    fout.write("=== BISECTION METHOD ===\n")
    fout.write("Iter\t   a\t\t   b\t\t   c\t\t f(c)\n")
    for i, a1, b1, c, fa, fb, fc in bis_solver.iterations:
        fout.write(f"{i:<6}{a1:<12.8f}{b1:<12.8f}{c:<12.8f}{fc:<12.8f}\n")
    fout.write(f"Root (Bisection) = {root_bis:.6f}\n\n")

    # Regula Falsi
    fout.write("=== REGULA FALSI METHOD ===\n")
    fout.write("Iter\t   a\t\t   b\t\t   c\t\t f(c)\n")
    for i, a1, b1, c, fa, fb, fc in rf_solver.iterations:
        fout.write(f"{i:<6}{a1:<12.8f}{b1:<12.8f}{c:<12.8f}{fc:<12.8f}\n")
    fout.write(f"Root (Regula Falsi) = {root_rf:.6f}\n\n")

    # Newton-Raphson
    fout.write("=== NEWTON-RAPHSON METHOD ===\n")
    fout.write("Iter\t   x\t\t   f(x)\t\t f'(x)\t\t   x_new\n")
    for i, x, fx, fpx, x_new in iter_nr:
        fout.write(f"{i:<6}{x:<12.8f}{fx:<12.8f}{fpx:<12.8f}{x_new:<12.8f}\n")
    fout.write(f"Root (Newton-Raphson) = {root_nr:.6f}\n\n")

    # Comparison
    fout.write("=== COMPARISON ===\n")
    fout.write(f"Bisection iterations: {len(bis_solver.iterations)}\n")
    fout.write(f"Regula Falsi iterations: {len(rf_solver.iterations)}\n")
    fout.write(f"Newton-Raphson iterations: {len(iter_nr)}\n\n")

    best = min(
        [("Bisection", len(bis_solver.iterations)),
         ("Regula Falsi", len(rf_solver.iterations)),
         ("Newton-Raphson", len(iter_nr))],
        key=lambda x: x[1]
    )
    fout.write(f"Best Method (fastest convergence): {best[0]}\n")




# PROBLEM 2: Fixed Point Method
# f(x) = x^2 - 2x - 3 = 0
# Roots: x = 3, x = -1
# Choose rearrangement: g(x) = sqrt(2x + 3), converges to x = 3

from mylib import FixedPointMethod

# Define g(x) for f(x) = x^2 - 2x - 3
def g(x):
    return math.sqrt(2*x + 3)   # converges to root near 3

solver = FixedPointMethod(g, tol=1e-6, max_iter=50)
root = solver.solve(2.0, filename="output1.txt")
