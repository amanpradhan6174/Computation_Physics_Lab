# assgn15.py
# Name - Aman Pradhan
# Roll Number - 2311020
# Driver file for assignment: uses mylib.ShootingSolver and mylib.HeatEquationSolver

import math
import matplotlib.pyplot as plt
from mylib import SimpsonRK4, RK4Integrator, ShootingSolver, HeatEquationSolver

# File setup
output_file = "output.txt"
process_file = "process.txt"
open(output_file, "w").close()
open(process_file, "w").close()


alpha = 0.01
Ta = 20.0
L = 10.0
T0 = 40.0
TL = 200.0

# we'll use an RK4Integrator instance
rk4int = RK4Integrator(func=None, name="RK4 for shooting")  # we'll replace func per use in ShootingSolver
shoot_solver = ShootingSolver(alpha=alpha, Ta=Ta, L=L, T0=T0, TL=TL, integrator=rk4int)

# small wrapper: set rk4int.func to shooting's system (the integrator stores func in instance variable)
rk4int.func = shoot_solver._system

# choose step h (coarse to moderate)
h = 0.05   # step size (produces 200 steps for L=10) - RK4 points will be visible
with open(process_file, "a") as pf:
    pf.write("\nStarting Shooting method for BVP\n")

slope, X_sol, T_sol = shoot_solver.solve(h=h, output_file=output_file, process_file=process_file, tol=1e-6, max_iter=40)

# find x for T=100
target_T = 100.0
x_at_100 = shoot_solver.find_x_for_temperature(X_sol, T_sol, target_T)

# write final summary
with open(output_file, "a") as of:
    of.write("\nBVP Shooting (RK4) result:\n")
    of.write(f"alpha = {alpha}, Ta = {Ta}, L = {L}\n")
    of.write(f"Boundary: T(0)={T0}, T(L)={TL}\n")
    of.write(f"Found initial slope s = {slope:.8f}\n")
    if x_at_100 is not None:
        of.write(f"Temperature T={target_T} occurs at x = {x_at_100:.6f}\n")
    else:
        of.write(f"Temperature T={target_T} not found in integration grid.\n")

# plot and save
ShootingSolver.plot_solution(X_sol, T_sol, target_temp=target_T, savefile="shooting_solution.png")


L2 = 2.0
nx = 100        # nodes between 0..nx -> dx = 2/nx (nx should be large enough)
dx = L2 / nx
dt = 0.0001     # choose dt so that r = dt/dx^2 is less than 0.5 (here dx^2 = (2/nx)^2)
r = dt / (dx*dx)

# safety: enforce small dt if r > 0.4
if r > 0.4:
    dt = 0.4 * dx * dx
    r = dt / (dx*dx)

t_final = 0.05   # final time to observe diffusion
peak = 300.0

# snapshot times to plot (start, small times, and final)
snapshot_times = [0.0, 0.002, 0.005, 0.01, 0.02, t_final]

heat_solver = HeatEquationSolver(L=L2, nx=nx, dt=dt, t_final=t_final, peak_value=peak)
snapshots = heat_solver.solve_and_record(output_file, process_file, snapshot_times=snapshot_times)

# save plot
HeatEquationSolver.plot_snapshots(snapshots, savefile="heat_equation_profiles.png")
