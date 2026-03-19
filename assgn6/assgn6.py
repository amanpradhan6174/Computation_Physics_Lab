# Name - Aman Pradhan 
# Roll Number - 2311020

# assgn6.py
import math
from mylib import BisectionMethod, RegulaFalsiMethod


# Problem 1

# Define the function for Problem 1
def f(x):
    return math.log(x / 2.0) - math.sin(5 * x / 2.0)

# Helper function: write iterations and final root to file
def write_iterations(fout, title, iterations, root):
    fout.write(f"{title} Iterations:\n")
    fout.write("iter\t   a\t\t   b\t\t   c\t\t f(a)\t\t f(b)\t\t f(c)\n")
    fout.write("-" * 80 + "\n")

    for (i, a, b, c, fa, fb, fc) in iterations:
        fout.write(f"{i:<4d}\t{a:.6f}\t{b:.6f}\t{c:.6f}\t{fa:.6f}\t{fb:.6f}\t{fc:.6f}\n")

    fout.write(f"\nFinal Root = {root:.8f}\n\n\n")

def main():
    # Initial bracket for problem 1
    a, b = 1.5, 3.0
    tol = 1e-6

    # Solve with Bisection
    bisect_solver = BisectionMethod(f, a, b, tol=tol)
    root_bisect = bisect_solver.solve()

    # Solve with Regula Falsi
    rf_solver = RegulaFalsiMethod(f, a, b, tol=tol)
    root_rf = rf_solver.solve()

    # Save results to output1.txt
    with open("output1.txt", "w") as fout:
        write_iterations(fout, "Bisection Method", bisect_solver.iterations, root_bisect)
        write_iterations(fout, "Regula Falsi Method", rf_solver.iterations, root_rf)



# Problem 2


# Function for Problem 2
def f(x):
    return -x - math.cos(x)

# Expand bracket outward until f(a) and f(b) have opposite signs
def expand_bracket(f, a, b, step=0.5, max_iter=50):
    for _ in range(max_iter):
        if f(a) * f(b) < 0:
            return a, b
        a -= step
        b += step
    raise ValueError("No root found within expanded range.")

# Write iterations for problem 2
def write_iterations(fout, title, iterations, root, a, b):
    fout.write(f"{title} Results:\n")
    fout.write(f"Confirmed bracket: [{a:.6f}, {b:.6f}]\n")
    fout.write("iter\t   a\t\t   b\t\t   c\t\t f(a)\t\t f(b)\t\t f(c)\n")
    fout.write("-" * 80 + "\n")

    for (i, a, b, c, fa, fb, fc) in iterations:
        fout.write(f"{i:<4d}\t{a:.6f}\t{b:.6f}\t{c:.6f}\t{fa:.6f}\t{fb:.6f}\t{fc:.6f}\n")

    fout.write(f"\nFinal Root = {root:.8f}\n\n\n")

def main():
    # Initial guess for Problem 2
    a, b = 2.0, 4.0
    tol = 1e-6

    # Expand until root is bracketed
    a, b = expand_bracket(f, a, b)

    # Solve with Bisection
    bisect_solver = BisectionMethod(f, a, b, tol=tol)
    root_bisect = bisect_solver.solve()

    # Solve with Regula Falsi
    rf_solver = RegulaFalsiMethod(f, a, b, tol=tol)
    root_rf = rf_solver.solve()

    # Save results to output2.txt
    with open("output2.txt", "w") as fout:
        write_iterations(fout, "Bisection Method", bisect_solver.iterations, root_bisect, a, b)
        write_iterations(fout, "Regula Falsi Method", rf_solver.iterations, root_rf, a, b)



# Run main

if __name__ == "__main__":
    main()
