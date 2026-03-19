# Assignment:1a (iterative equation and LCG random generator)
# Name : Aman Pradhan and Roll_No : 2311020
# Name: Aman Pradhan, RollNo: 2311020
# SectionA: Logistic map correlation plots
# SectionB: LCG random generator correlation plot

import matplotlib.pyplot as plt

# SECTION A: Logistic map random number generator
def logistic_map(seed, c, num_points):
    x_values = [seed]
    for _ in range(num_points - 1):
        x_next = c * x_values[-1] * (1 - x_values[-1])
        x_values.append(x_next)
    return x_values

# SECTION B: Linear Congruential Generator (LCG)
def lcg(seed, a, c, m, num_points):
    values = [seed]
    for _ in range(num_points - 1):
        next_val = (a * values[-1] + c) % m
        values.append(next_val / m)  # Normalize to [0,1]
    return values

def main():
    
        print("Name: Aman Pradhan, RollNo: 2311020\n\n")

        # SECTION A: Logistic Map
        seed = 0.1
        num_points = 1000
        c_values = [2.5, 3.2, 3.5, 3.8, 4.0]  # Five different c
        n_values = [5, 10, 15, 20]

        print("SECTION A: Logistic Map Correlation\n")
        for c in c_values:
            x_vals = logistic_map(seed, c, num_points)
            print(f"\nFor c = {c}:\n")
            for n in n_values:
                plt.figure()
                plt.scatter(x_vals[:-n], x_vals[n:], s=5)
                plt.xlabel(f"x_i")
                plt.ylabel(f"x_(i+{n})")
                plt.title(f"Logistic Map Correlation (c={c}, n={n})")
                plt.savefig(f"logistic_c{c}_n{n}.png", dpi=150)
                plt.close()
                print(f"  Correlation plot logistic_c{c}_n{n}.png\n")

        # SECTION B: LCG
        print("\nSECTION B: LCG Random Generator Correlation (k=5)\n")
        a, c_lcg, m = 1103515245, 12345, 32768
        seed_lcg = 1
        k = 5
        num_points_lcg = 1000

        lcg_vals = lcg(seed_lcg, a, c_lcg, m, num_points_lcg)

        plt.figure()
        plt.scatter(lcg_vals[:-k], lcg_vals[k:], s=5, color='red')
        plt.xlabel("X_i")
        plt.ylabel(f"X_(i+{k})")
        plt.title(f"LCG Correlation (k={k})")
        plt.savefig(f"lcg_correlation_k{k}.png", dpi=150)
        plt.close()

        print(f"  Correlation plot saved: lcg_correlation_k{k}.png\n")


if __name__ == "__main__":
    main()
