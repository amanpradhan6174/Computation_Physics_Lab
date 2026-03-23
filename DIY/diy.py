# DIY Project: Adaptive RKF45 vs RK4 for Spring–Mass–Damper System
# Author: Aman Pradhan
# Roll No: 231020
# Reproduction of Ng Ching Kok & Shazirawati Mohd Puzi (2022)
# Equation: m*x'' + a*x' + k*x = F0*sin(ωt)
# Purpose: Compare RK4 (fixed step) with RKF45 (adaptive step)
import math
import matplotlib.pyplot as plt
import time

# Given System Parameters
m, a, k = 1.0, 0.5, 4.0
F0, omega = 5.0, 2.0
x0, v0 = 0.0, 0.0
t0, tmax = 0.0, 20.0
tol = 1e-6

# Differential Equation System
# y = [x, x']
def f(t, y):
    x, v = y
    dx = v
    dv = (F0 * math.sin(omega * t) - a * v - k * x) / m
    return [dx, dv]

# Classical RK4 (Fixed Step)
def rk4(f, y0, t0, tmax, h):
    t = t0
    y = y0[:]
    T, X, V = [t], [y[0]], [y[1]]
    while t < tmax - 1e-12:
        if t + h > tmax:
            h = tmax - t
        # Compute 4 intermediate slopes
        k1 = f(t, y)
        k2 = f(t + h/2, [y[i] + h*k1[i]/2 for i in range(2)])
        k3 = f(t + h/2, [y[i] + h*k2[i]/2 for i in range(2)])
        k4 = f(t + h, [y[i] + h*k3[i] for i in range(2)])
        # Update solution
        y = [y[i] + h*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6 for i in range(2)]
        t += h
        T.append(t); X.append(y[0]); V.append(y[1])
    return T, X, V

# Adaptive RKF45 Method
def rkf45(f, y0, t0, tmax, h_init, tol):
    # Fehlberg coefficients
    a2,a3,a4,a5,a6 = 1/4,3/8,12/13,1.0,1/2
    b21=1/4
    b31,b32=3/32,9/32
    b41,b42,b43=1932/2197,-7200/2197,7296/2197
    b51,b52,b53,b54=439/216,-8,3680/513,-845/4104
    b61,b62,b63,b64,b65=-8/27,2,-3544/2565,1859/4104,-11/40
    c=[16/135,0,6656/12825,28561/56430,-9/50,2/55]
    cs=[25/216,0,1408/2565,2197/4104,-1/5,0]

    t, y, h = t0, y0[:], h_init
    T, X, V, H = [t], [y[0]], [y[1]], [h]

    while t < tmax - 1e-12:
        if t + h > tmax:
            h = tmax - t
        # Calculate 6 intermediate slopes
        k1=f(t,y)
        k2=f(t+a2*h,[y[i]+h*b21*k1[i] for i in range(2)])
        k3=f(t+a3*h,[y[i]+h*(b31*k1[i]+b32*k2[i]) for i in range(2)])
        k4=f(t+a4*h,[y[i]+h*(b41*k1[i]+b42*k2[i]+b43*k3[i]) for i in range(2)])
        k5=f(t+a5*h,[y[i]+h*(b51*k1[i]+b52*k2[i]+b53*k3[i]+b54*k4[i]) for i in range(2)])
        k6=f(t+a6*h,[y[i]+h*(b61*k1[i]+b62*k2[i]+b63*k3[i]+b64*k4[i]+b65*k5[i]) for i in range(2)])

        y4=[y[i]+h*sum(cs[j]*[k1,k2,k3,k4,k5,k6][j][i] for j in range(6)) for i in range(2)]
        y5=[y[i]+h*sum(c[j]*[k1,k2,k3,k4,k5,k6][j][i] for j in range(6)) for i in range(2)]

        err = max(abs(y5[i]-y4[i]) for i in range(2))
        if err <= tol:
            t += h; y = y5
            T.append(t); X.append(y[0]); V.append(y[1]); H.append(h)
        s = 2 if err == 0 else 0.9*(tol/err)**0.25
        h = h * min(max(s, 0.1), 4.0)
    return T, X, V, H

# Analytical Solution
p = k - omega**2
det = p*p + (a*omega)**2
A = (F0*p)/det
B = (-F0*a*omega)/det
alpha = -a/2
beta = math.sqrt(4*k - a*a)/2
C = x0 - B
D = -(A*omega + alpha*C)/beta

def x_exact(t):
    xh = math.exp(alpha*t)*(C*math.cos(beta*t)+D*math.sin(beta*t))
    xp = A*math.sin(omega*t)+B*math.cos(omega*t)
    return xh + xp

def dxdt_exact(t):
    xh_dot = math.exp(alpha*t)*(alpha*(C*math.cos(beta*t)+D*math.sin(beta*t))
               + beta*(-C*math.sin(beta*t)+D*math.cos(beta*t)))
    xp_dot = A*omega*math.cos(omega*t)-B*omega*math.sin(omega*t)
    return xh_dot + xp_dot

# Run Both Solvers
start = time.time()
T_rk4, X_rk4, V_rk4 = rk4(f, [x0,v0], t0, tmax, 0.05)
T_rkf, X_rkf, V_rkf, H_rkf = rkf45(f, [x0,v0], t0, tmax, 0.5, tol)
end = time.time()

print(f"\nSimulation complete in {end-start:.4f} seconds")
print(f"Total RK4 steps   : {len(T_rk4)}")
print(f"Total RKF45 steps : {len(T_rkf)}\n")
# Save Table (RK4 vs RKF45 vs Exact)
table_data = []
for i, t in enumerate(T_rk4):
    x_rk4 = X_rk4[i]
    idx = min(range(len(T_rkf)), key=lambda j: abs(T_rkf[j]-t))
    x_rkf = X_rkf[idx]
    x_true = x_exact(t)
    table_data.append((i+1, t, x_rk4, x_rkf, x_true))

with open("table1_results.txt", "w") as f:
    f.write("Table 1: RK4 and RKF45 Approximation Results\n")
    f.write("{:<8}{:<10}{:<15}{:<15}{:<15}\n".format("Iter","Time","RK4","RKF45","Analytical"))
    for row in table_data:
        f.write("{:<8}{:<10.2f}{:<15.6f}{:<15.6f}{:<15.6f}\n".format(*row))


# Plot Results
# 1. Error comparison (matched to thesis Figure)
# Uniform time grid 0, 0.5, 1.0, ... , 10
times_uniform = [round(0.5*i, 1) for i in range(int(tmax/0.5)+1)]

# RK4 with fixed h = 0.5 already gives values at these points
exact_vals = [x_exact(t) for t in times_uniform]
err_rk4 = [abs(X_rk4[i] - exact_vals[i]) for i in range(len(times_uniform))]

#Interpolate RKF45 onto the same uniform grid
def interp_linear(x_list, y_list, x):
    if x <= x_list[0]:
        return y_list[0]
    if x >= x_list[-1]:
        return y_list[-1]
    for i in range(len(x_list)-1):
        if x_list[i] <= x <= x_list[i+1]:
            x0, x1 = x_list[i], x_list[i+1]
            y0, y1 = y_list[i], y_list[i+1]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return y_list[-1]

X_rkf_interp = [interp_linear(T_rkf, X_rkf, t) for t in times_uniform]
err_rkf = [abs(X_rkf_interp[i] - exact_vals[i]) for i in range(len(times_uniform))]
# Plot as in the PDF 
plt.figure(figsize=(7,4))
plt.plot(times_uniform, err_rk4, 'b-', label='RK4 (h=0.05)')
plt.plot(times_uniform, err_rkf, 'g-', label='RKF45 (ode45)')
plt.xlabel("Time (s)")
plt.ylabel("Absolute Error")
plt.title("Error of function ode45 and Runge Kutta Order 4 (h=0.05)")
plt.legend()
plt.grid(True)
plt.savefig("figure1_error_matched_pdf.png", dpi=300)

# 2. Analytical vs Numerical
times = [i*0.5 for i in range(int(tmax/0.5)+1)]
plt.figure()
plt.plot(times, [x_exact(t) for t in times], 'bo', label='Analytical x(t)')
plt.plot(times, [dxdt_exact(t) for t in times], 'r*', label='Analytical dx/dt')
plt.plot(T_rkf, X_rkf, 'm-', label='Numerical x(t)')
plt.plot(T_rkf, V_rkf, 'y-', label='Numerical dx/dt')
plt.xlabel("Time (s)"); plt.ylabel("Response")
plt.title("Analytical vs RKF45 Numerical Solution")
plt.legend(); plt.grid(True)
plt.savefig("figure2_rkf45.png", dpi=300)

# 3. Step size adaptation
plt.figure()
plt.plot(T_rkf, H_rkf, 'c-')
plt.xlabel("Time (s)"); plt.ylabel("Step size h")
plt.title("Adaptive Step Size in RKF45")
plt.grid(True)
plt.savefig("figure3_h_vs_t.png", dpi=300)

# 4. All solutions comparison
plt.figure()
plt.plot(T_rk4, [x_exact(t) for t in T_rk4], 'b-.', label='Exact')
plt.scatter(T_rk4, X_rk4, c='green', marker='D', label='RK4')
plt.scatter(T_rkf, X_rkf, c='red', marker='o', label='RKF45')
plt.xlabel("Time (s)"); plt.ylabel("x(t)")
plt.title("Comparison of RK4, RKF45, and Exact Solutions")
plt.legend(); plt.grid(True)
plt.savefig("figure4_comparison.png", dpi=300)

print("Max RK4 error:", max(err_rk4))
print("Max RKF45 error:", max(err_rkf))
