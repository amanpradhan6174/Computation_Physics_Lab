class LCG:
    def __init__(self, seed=1):
        self.a = 1103515245
        self.c = 12345
        self.m = 32768
        self.state = seed

    def lcg(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

class GaussJordanSolver:
    def __init__(self, matrix):
        self.mat = [row[:] for row in matrix]
        self.n = len(matrix)
        self.log = []

    def print_matrix(self, msg):
        self.log.append(msg)
        for row in self.mat:
            line = " ".join(f"{x:8.4f}" for x in row)
            self.log.append(line)
        self.log.append('')  # Blank line

    def solve(self):
        # Step 1: Special pivot for the first column only
        max_row = 0
        # Find among rows 0,1,2 the row where abs(mat[.][0]) is largest for column 0
        for r in range(1, self.n):
            if abs(self.mat[r]) > abs(self.mat[max_row]):
                max_row = r
        # If a11 is 0, swap with row of greatest value in first column
        if self.mat == 0:
            self.mat, self.mat[max_row] = self.mat[max_row], self.mat
            self.print_matrix(f"Swapped row 1 with row {max_row + 1} for first pivot (first column):")
        else:
            self.print_matrix("No swap needed for 1st pivot (first column):")

        # Standard Gauss-Jordan elimination steps
        for i in range(self.n):
            pivot = self.mat[i][i]
            if pivot == 0:
                raise ValueError("Matrix is singular!")
            # Make leading coefficient 1
            self.mat[i] = [x / pivot for x in self.mat[i]]
            self.print_matrix(f"R{i+1} made leading 1:")

            # Eliminate other rows
            for j in range(self.n):
                if j != i:
                    factor = self.mat[j][i]
                    self.mat[j] = [
                        self.mat[j][k] - factor * self.mat[i][k]
                        for k in range(self.n + 1)
                    ]
            self.print_matrix(f"Eliminated column {i+1} in all other rows:")

        solution = [self.mat[i][-1] for i in range(self.n)]
        self.log.append("Final Solution: " + " ".join(f"x{i+1}={solution[i]:.4f}" for i in range(self.n)))
        return solution

    def get_log(self):
        return "\n".join(self.log)
