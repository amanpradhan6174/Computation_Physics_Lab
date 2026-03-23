# Name - Aman Pradhan
# Roll Number - 2311020
# assgn14.py (with markers at RK4 points)
# Assignment 14 Driver File: Calls SimpsonRK4 and RK4Integrator


import math
import matplotlib.pyplot as plt
from mylib import SimpsonRK4, RK4Integrator

#  File Setup
output_file = "output.txt"
process_file = "process.txt"

# Clear files before writing
open(output_file, "w").close()
open(process_file, "w").close()


# Problem 1: RK4 for dy/dx = (x+y)^2
# Analytical: y = tan(x + π/4) - x


def rhs_problem1(y_list, x):
    y = y_list[0]
    return [(x + y) ** 2]

def analytical_problem1(x):
    return math.tan(x + math.pi / 4.0) - x

x0, y0 = 0.0, [1.0]
x_end = math.pi / 5.0
step_sizes = [0.1, 0.25, 0.45]

rk4_problem1 = RK4Integrator(rhs_problem1, "RK4 Problem 1")

# Plot analytical solution
X_exact = [x0 + i*0.001 for i in range(int((x_end-x0)/0.001)+1)]
Y_exact = [analytical_problem1(x) for x in X_exact]
plt.figure(figsize=(8,6))
plt.plot(X_exact, Y_exact, 'k--', label='Analytical')

# Compute and plot RK4 for each step size with markers
for h in step_sizes:
    X_vals, Y_vals = rk4_problem1.integrate_and_write(y0, x0, x_end, h, output_file, process_file)
    plt.plot(X_vals, [y[0] for y in Y_vals], marker='o', markersize=4, linestyle='-', label=f'h={h}')

plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('RK4 Solution vs Analytical')
plt.legend()
plt.grid(True)
plt.savefig('rk4_problem1.png', dpi=300)
plt.close()


# Problem 2: Damped SHO
# x¨ + μx˙ + ω²x = 0

def rhs_damped_sho(mu, omega):
    def f(y, t):
        x = y[0]
        v = y[1]
        dxdt = v
        dvdt = -mu * v - (omega**2) * x
        return [dxdt, dvdt]
    return f

def energy_sho(x, v, m=1.0, k=1.0):
    return 0.5*m*v*v + 0.5*k*x*x

mu, omega = 0.15, 1.0
t0, tf, dt = 0.0, 40.0, 0.05
y0_sho = [1.0, 0.0]

rk4_sho = RK4Integrator(rhs_damped_sho(mu, omega), "RK4 Damped SHO")
T_vals, Y_vals = rk4_sho.integrate_and_write(y0_sho, t0, tf, dt, output_file, process_file)

x_vals = [y[0] for y in Y_vals]
v_vals = [y[1] for y in Y_vals]
E_vals = [energy_sho(x_vals[i], v_vals[i]) for i in range(len(T_vals))]

#  Plot 1: x vs v 
plt.figure(figsize=(6,5))
plt.plot(x_vals, v_vals, 'b-', marker='o', markersize=3)
plt.xlabel('x')
plt.ylabel('v')
plt.title('Damped SHO: Phase Space (x vs v)')
plt.grid(True)
plt.savefig('sho_x_vs_v.png', dpi=300)
plt.close()

#  Combined plot: x(t), v(t), E(t) 
plt.figure(figsize=(7,10))

plt.subplot(3,1,1)
plt.plot(T_vals, x_vals, 'r-', marker='o', markersize=2)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Damped SHO: x vs t')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(T_vals, v_vals, 'g-', marker='o', markersize=2)
plt.xlabel('t')
plt.ylabel('v(t)')
plt.title('Damped SHO: v vs t')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(T_vals, E_vals, 'm-', marker='o', markersize=2)
plt.xlabel('t')
plt.ylabel('E(t)')
plt.title('Damped SHO: Energy vs Time')
plt.grid(True)

plt.tight_layout()
plt.savefig('sho_combined_xt_vt_Et.png', dpi=300)
plt.close()