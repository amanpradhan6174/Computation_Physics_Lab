#Name-Aman Pradhan
#Roll no- 2311020
from mylib import GaussJordanSolver

# Read matrix from file
input_file1 = 'assgn_2a_matva.txt'

matrix = []
with open(input_file1, 'r') as f:
    for line in f:
        if line.strip():
            matrix.append([float(x) for x in line.strip().split()])

solver = GaussJordanSolver(matrix)
solution = solver.solve()

# Print to console as well
print(solver.get_log())

# Read matrix from file
input_file2 = 'assgn_2a_matvb.txt'

matrix = []
with open(input_file2, 'r') as f:
    for line in f:
        if line.strip():
            matrix.append([float(x) for x in line.strip().split()])

solver = GaussJordanSolver(matrix)
solution = solver.solve()

# Print to console as well
print(solver.get_log())
