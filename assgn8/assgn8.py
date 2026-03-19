# AASSIGNMENT-8
# NAME - Aman Pradhan
# Roll Number - 2311020
import math
from mylib import FixedPointMethodMultivar, NewtonRaphsonMultivar

# ---------------- Problem Definition ---------------- #
# System of equations:
# x1^2 + x2 = 37
# x1 - x2^2 = 5
# x1 + x2 + x3 = 3

# Fixed-Point functions
def g1(x): return math.sqrt(37 - x[1])
def g2(x): return math.sqrt(x[0] - 5)
def g3(x): return 3 - (x[0] + x[1])

# Newton-Raphson functions
def f1(x): return x[0]**2 + x[1] - 37
def f2(x): return x[0] - x[1]**2 - 5
def f3(x): return x[0] + x[1] + x[2] - 3

# Initial guess
x0 = [5.0, 2.0, -4.0]

# ---------------- Solve with Fixed-Point ---------------- #
fp_solver = FixedPointMethodMultivar([g1, g2, g3])
fp_solution, fp_history = fp_solver.iterate(x0, filename="output.txt", return_history=True)

# ---------------- Solve with Newton-Raphson ---------------- #
nr_solver = NewtonRaphsonMultivar([f1, f2, f3], tol=1e-8)
nr_solution, nr_history = nr_solver.iterate(x0, filename="output.txt", return_history=True, append=True)

# ---------------- Convergence Rate Comparison ---------------- #
fp_iterations = len(fp_history)
nr_iterations = len(nr_history)

convergence_info = (
    "\n---------------- Convergence Rate Comparison ----------------\n"
    f"Fixed-Point Method converged in {fp_iterations} iterations\n"
    f"Newton-Raphson Method converged in {nr_iterations} iterations\n"
)

# Print to console
print(convergence_info)

# Append to output.txt
with open("output.txt", "a") as f:
    f.write(convergence_info)

# ---------------- Print Final Solutions ---------------- #
print("Final Fixed-Point solution: ", " ".join(f"{val:.6f}" for val in fp_solution))
print("Final Newton-Raphson solution:", " ".join(f"{val:.6f}" for val in nr_solution))
