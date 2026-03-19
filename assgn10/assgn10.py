# assgn10.py
# Name - Aman Pradhan
# Roll Number - 2311020

import math
from mylib import MidpointIntegration, TrapezoidalIntegration


# Define the three problem functions and their f''(x)

def f1(x): return 1 / x
def f1_dd(x): return 2 / (x ** 3)

def f2(x): return x * math.cos(x)
def f2_dd(x): return -2 * math.sin(x) - x * math.cos(x)

def f3(x): return x * math.atan(x)
def f3_dd(x): return (2 * x) / ((1 + x ** 2) ** 2)


# Analytical (exact) results from problem statement

I1_exact = math.log(2)              # Integral of 1/x from 1 to 2
I2_exact = (math.pi / 2) - 1        # Integral of x*cos(x) from 0 to pi/2
I3_exact = (math.pi / 4) - 0.5      # Integral of x*atan(x) from 0 to 1

# Number of subintervals (N values) to use
N_values = [4, 8, 15, 20]

# Clear previous results before writing new ones

open("output.txt", "w", encoding="utf-8").close()
open("process.txt", "w", encoding="utf-8").close()



# Create solver objects for each integral and method

# Using plain ASCII text names (no symbols like ∫, π)
mid1 = MidpointIntegration(f1, f1_dd, 1, 2, I1_exact, N_values, "Integral_1: from 1 to 2 of 1/x dx")
trap1 = TrapezoidalIntegration(f1, f1_dd, 1, 2, I1_exact, N_values, "Integral_1: from 1 to 2 of 1/x dx")

mid2 = MidpointIntegration(f2, f2_dd, 0, math.pi/2, I2_exact, N_values, "Integral_2: from 0 to pi/2 of x*cos(x) dx")
trap2 = TrapezoidalIntegration(f2, f2_dd, 0, math.pi/2, I2_exact, N_values, "Integral_2: from 0 to pi/2 of x*cos(x) dx")

mid3 = MidpointIntegration(f3, f3_dd, 0, 1, I3_exact, N_values, "Integral_3: from 0 to 1 of x*atan(x) dx")
trap3 = TrapezoidalIntegration(f3, f3_dd, 0, 1, I3_exact, N_values, "Integral_3: from 0 to 1 of x*atan(x) dx")


# Run computations and write results

mid1.compute_and_write("output.txt", "process.txt")
trap1.compute_and_write("output.txt", "process.txt")

mid2.compute_and_write("output.txt", "process.txt")
trap2.compute_and_write("output.txt", "process.txt")

mid3.compute_and_write("output.txt", "process.txt")
trap3.compute_and_write("output.txt", "process.txt")

print(" Computation complete. Check 'output.txt' for results and 'process.txt' for step details.")
