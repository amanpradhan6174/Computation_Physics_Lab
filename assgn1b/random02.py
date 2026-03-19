# Assignment1: question no. 3 & 4
# Name = Aman Pradhan and Roll No. 2311020
import matplotlib.pyplot as plt
import numpy as np
from mylib import LCG

seed = 10
lcg = LCG(seed)

# to generate uniform [0,1)
def lcg_rand01():
    return lcg.lcg() / lcg.m

#  Question 3: Estimate pi
N_values = np.arange(20, 2001)
pi_estimates = []

for N in N_values:
    inside = 0
    for _ in range(N):
        x = lcg_rand01()
        y = lcg_rand01()
        if x**2 + y**2 <= 1.0:
            inside += 1
    # pi ≈ 4 × (inside/total)
    pi_estimates.append(4 * inside / N)
print('Avarage value of pi:', np.mean(pi_estimates))

plt.scatter(N_values, pi_estimates,s= 3, label="LCG Estimate")
plt.axhline(np.pi, color='r', linestyle='--', label="Actual pi")
plt.xlabel("Number of Throws")
plt.ylabel("Estimated pi")
plt.title(" Estimation of pi by LCG")
plt.legend()
plt.savefig("pi_estimation_lcg.png")

# Question 4: Exponential Distribution via Inverse Transform
# q(y) = exp(-y)
# Inverse: y = -ln(x) (for x∈[0,1), but avoid x=0)

N_exp = 5000
exponentials = []

for _ in range(N_exp):
    u = lcg_rand01()
    # Avoid log(0) by shifting in case u==0 (rare)
    if u == 0:
        u = 1e-10
    y = -np.log(u)
    exponentials.append(y)

plt.hist(exponentials, bins=40, color="skyblue", edgecolor="black")
plt.xlabel("x")
plt.ylabel("Count")
plt.title("Histogram: Exponential Distribution (exp(-x)), LCG")
plt.savefig("exponential_histogram_lcg.png")    

# answer in terminal : Avarage value of pi: 3.1434058138366248