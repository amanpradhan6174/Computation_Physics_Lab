# Assignment -13
# Name - Aman Pradhan
# Roll Number - 2311020

import math
import matplotlib.pyplot as plt
from mylib import ForwardEulerSolver, PredictorCorrectorSolver

# Plot Comparison Function
def plot_comparison(data_euler, data_pc, exact_func, x0, h, N, title, savefile=None):
    x_e = [p[0] for p in data_euler]
    y_e = [p[1] for p in data_euler]
    x_pc = [p[0] for p in data_pc]
    y_pc = [p[1] for p in data_pc]
    x_exact = [x0 + i * h for i in range(N + 1)]
    y_exact = [exact_func(x) for x in x_exact]

    plt.figure(figsize=(6, 4))
    plt.plot(x_exact, y_exact, linewidth=1.2, label='Exact')
    plt.scatter(x_e, y_e, s=20, marker='o', label='Fwd Euler')
    plt.scatter(x_pc, y_pc, s=20, marker='s', facecolors='none',
                edgecolors='orange', label='Predictor-Corrector')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150)
    plt.show()


# Files for output 
process_file = "process.txt"
output_file = "output.txt"
open(process_file, "w").close()
open(output_file, "w").close()


# Problem 1: dy/dx = y − x², y(0)=0, y(x)=x²+2x+2−2e^x,  x∈[0,2]

def f1(x, y): return y - x**2
def y_exact1(x): return x**2 + 2*x + 2 - 2*math.exp(x)

x0, y0, h, N = 0, 0, 0.1, int((2 - 0) / 0.1)

fe1 = ForwardEulerSolver(f1, x0, y0, h, N)
pc1 = PredictorCorrectorSolver(f1, x0, y0, h, N)
data_euler1 = fe1.write_output(process_file, output_file, y_exact1, "dy/dx = y - x^2")
data_pc1 = pc1.write_output(process_file, output_file, y_exact1, "dy/dx = y - x^2")
plot_comparison(data_euler1, data_pc1, y_exact1, x0, h, N,
                "dy/dx = y - x²", "graph1.png")


# Problem 2: dy/dx = (x + y)², y(0)=1, tan⁻¹(x+y)=x+π/4,  x∈[0,π/5]

def f2(x, y): return (x + y)**2
def y_exact2(x): return math.tan(x + math.pi/4) - x

x0, y0, h, N = 0, 1, 0.1, int((math.pi/5 - 0) / 0.1)

fe2 = ForwardEulerSolver(f2, x0, y0, h, N)
pc2 = PredictorCorrectorSolver(f2, x0, y0, h, N)
data_euler2 = fe2.write_output(process_file, output_file, y_exact2, "dy/dx = (x + y)^2")
data_pc2 = pc2.write_output(process_file, output_file, y_exact2, "dy/dx = (x + y)^2")
plot_comparison(data_euler2, data_pc2, y_exact2, x0, h, N,
                "dy/dx = (x + y)²", "graph2.png")

print(" Results saved in process.txt, output.txt, and graph1.png, graph2.png")
