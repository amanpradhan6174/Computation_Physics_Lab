
# Name - Aman Pradhan
# Roll Number - 2311020
# Assignment 16: Lagrange Interpolation & Curve Fitting

import matplotlib.pyplot as plt
from mylib import LagrangeSolver, CurveFitter

#  File Setup
output_file = "output.txt"
process_file = "process.txt"
open(output_file, "w").close()
open(process_file, "w").close()

# -------------------------------------------------------
# Problem 1: Lagrange Interpolation
# -------------------------------------------------------

x_vals = [2, 3, 5, 8, 12]
y_vals = [10, 15, 25, 40, 60]
x0 = 6.7

solver = LagrangeSolver(x_vals, y_vals)
y_est, details = solver.interpolate(x0)
solver.plot(x0, filename="problem1.png")

# Write Process Details
with open(process_file, "a") as pf:
    pf.write("Problem 1: Lagrange Interpolation\n")
    for line in details:
        pf.write(line + "\n")

# Write Output Summary
with open(output_file, "a") as of:
    of.write(f"Problem 1: Estimated y(6.7) = {y_est:.6f}\n")

# -------------------------------------------------------
# Problem 2: Curve Fitting (Power Law & Exponential)
# Pearson correlation coefficient r^2 is used for goodness of fit.
# -------------------------------------------------------

x2 = [2.5, 3.5, 5.0, 6.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.5]
y2 = [13.0, 11.0, 8.5, 8.2, 7.0, 6.2, 5.2, 4.8, 4.6, 4.3]

fitter = CurveFitter(x2, y2)

# Power Law Fit 
a_p, b_p, r2_p, sigma_p, details_p = fitter.fit_power()
fitter.plot_power(filename="problem2_power.png")

# Exponential Fit 
a_e, b_e, r2_e, sigma_e, details_e = fitter.fit_exponential()
fitter.plot_exponential(filename="problem2_exp.png")

#  Write Detailed Process 
with open(process_file, "a") as pf:
    pf.write("\nProblem 2: Curve Fitting (Power & Exponential)\n")
    pf.write("--- Power Law Details ---\n")
    for line in details_p:
        pf.write(line + "\n")
    pf.write("--- Exponential Details ---\n")
    for line in details_e:
        pf.write(line + "\n")

#  Write Summary with sigma
with open(output_file, "a") as of:
    of.write("\nProblem 2: Model Comparison\n")
    of.write(
        f"Power law: y = {a_p:.6f} * x^{b_p:.6f}, "
        f"r^2 = {r2_p:.6f}, sigma = {sigma_p:.6f}\n"
    )
    of.write(
        f"Exponential: y = {a_e:.6f} * exp(-{b_e:.6f}x), "
        f"r^2 = {r2_e:.6f}, sigma = {sigma_e:.6f}\n"
    )
    better = "Power law" if r2_p > r2_e else "Exponential"
    of.write(f"Better fit based on r^2: {better}\n")

