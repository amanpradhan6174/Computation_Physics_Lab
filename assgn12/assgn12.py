
# Name  : Aman Pradhan
# Roll No : 2311020
# Assignment 12 — Numerical Integration
# Comparison of Simpson's Rule (Manual N) and Gaussian Quadrature


from mylib import GaussianQuadrature, SimpsonWithoutNFinding
import math

out1, proc1 = "output12.txt", "process12.txt"
open("output12.txt", "w").close()
open("process12.txt", "w").close()
# Problem 1 
def f1(x):
    return (x**2) / (1 + x**4)

exact1 = 0.487495494
N1 = 16

simp1 = SimpsonWithoutNFinding(f1, -1, 1, exact1, "x^2/(1+x^4) from -1 to 1")
simp_result1, simp_err1 = simp1.compute_and_write(out1, proc1, N1)

gq1 = GaussianQuadrature(f1, -1, 1, 16, exact1, "x^2/(1+x^4) from -1 to 1")
gq1_result, gq1_error = gq1.compute_and_write(out1, proc1)

#  Problem 2 
def f2(x):
    return math.sqrt(1 + x**4)

exact2 = 1.089429413
N2 = 16

simp2 = SimpsonWithoutNFinding(f2, 0, 1, exact2, "sqrt(1+x^4) from 0 to 1")
simp_result2, simp_err2 = simp2.compute_and_write(out1, proc1, N2)

gq2 = GaussianQuadrature(f2, 0, 1, 16, exact2, "sqrt(1+x^4) from 0 to 1")
gq2_result, gq2_error = gq2.compute_and_write(out1, proc1)

#  Comparison Table
with open(out1, "a") as f_out:
    f_out.write("\n" + "="*70 + "\nCOMPARISON TABLE\n" + "="*70 + "\n")
    f_out.write(f"{'Problem':<10}{'Method':<20}{'Approximation':<20}{'Error'}\n")
    f_out.write("-"*70 + "\n")

    f_out.write(f"{'1':<10}{'Simpson(N=16)':<20}{simp_result1:<20.9f}{simp_err1:.2e}\n")
    f_out.write(f"{'1':<10}{'Gaussian(n=16)':<20}{gq1_result:<20.9f}{gq1_error:.2e}\n\n")
    f_out.write(f"{'2':<10}{'Simpson(N=16)':<20}{simp_result2:<20.9f}{simp_err2:.2e}\n")
    f_out.write(f"{'2':<10}{'Gaussian(n=16)':<20}{gq2_result:<20.9f}{gq2_error:.2e}\n")
