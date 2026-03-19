# Name - Aman Pradhan
# Roll Number - 2311020

# Assignment 11 — Numerical Integration
# Uses classes from mylib.py:
#   - MidpointNfindingIntegration
#   - SimpsonIntegration
#   - MonteCarloIntegration


import math
from mylib import MidpointNfindingIntegration, SimpsonIntegration, MonteCarloIntegration


# Tolerances

tol6 = 1e-6   # for Midpoint & Simpson methods
tol4 = 1e-3   # for Monte Carlo integration (3-decimal accuracy)


# === Problem 1: Deterministic Integration ===

out1, proc1 = "output1.txt", "process1.txt"

# (a) f(x) = 1/x, ∫₁→₂ (1/x) dx = ln(2)
def f1(x): 
    return 1 / x

def f1_dd(x): 
    return 2 / (x ** 3)

def f1_4(x): 
    return 24 / (x ** 5)

exact1 = math.log(2)

MidpointNfindingIntegration(
    f1, f1_dd, 1, 2, exact1, "∫₁→₂ 1/x dx"
).compute_and_write(out1, proc1, tol6)

SimpsonIntegration(
    f1, f1_4, 1, 2, exact1, "∫₁→₂ 1/x dx"
).compute_and_write(out1, proc1, tol6)


# (b) f(x) = x cos(x), ∫₀→π/2 x cos(x) dx = π/2 − 1
def f2(x): 
    return x * math.cos(x)

def f2_dd(x): 
    return -2 * math.sin(x) - x * math.cos(x)

def f2_4(x): 
    return 4 * math.sin(x) + x * math.cos(x)

exact2 = math.pi / 2 - 1

MidpointNfindingIntegration(
    f2, f2_dd, 0, math.pi / 2, exact2, "∫₀→π/2 x cos(x) dx"
).compute_and_write(out1, proc1, tol6)

SimpsonIntegration(
    f2, f2_4, 0, math.pi / 2, exact2, "∫₀→π/2 x cos(x) dx"
).compute_and_write(out1, proc1, tol6)

# === Problem 2: Monte Carlo Integration ===

out2, proc2 = "output2.txt", "process2.txt"

# f(x) = sin²(x), ∫₋₁→₁ sin²(x) dx = 1 − (sin(2)/2)
def f3(x): 
    return math.sin(x) ** 2

exact3 = 1 - (math.sin(2) / 2)

# Initialize Monte Carlo solver
mc = MonteCarloIntegration(
    f3, -1, 1, exact3, lcg_seed=7, name="∫₋₁→₁ sin²(x) dx"
)

# Run simulation until tolerance met
estimate, n, err = mc.run_until_tol(
    tol_abs=tol4,
    output_file=out2,
    process_file=proc2,
    plot_file="mc_plot.png"
)


