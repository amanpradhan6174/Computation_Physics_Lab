# Assignment - 9
# Name - Aman Pradhan
# Roll Number - 2311020

from mylib import LaguerreDeflationSolver
open("output.txt", "w").close()
# P1(x) = x^4 - x^3 - 7x^2 + x + 6
coeffs1 = [1, -1, -7, 1, 6]
solver1 = LaguerreDeflationSolver(coeffs1)
solver1.solve("P1(x) = x^4 - x^3 - 7x^2 + x + 6")

# P2(x) = x^4 - 5x^2 + 4
coeffs2 = [1, 0, -5, 0, 4]
solver2 = LaguerreDeflationSolver(coeffs2)
solver2.solve("P2(x) = x^4 - 5x^2 + 4")

# P3(x) = 2x^5 - 19.5x^3 + 0.5x^2 + 13.5x - 4.5
coeffs3 = [2, 0, -19.5, 0.5, 13.5, -4.5]
solver3 = LaguerreDeflationSolver(coeffs3)
solver3.solve("P3(x) = 2x^5 - 19.5x^3 + 0.5x^2 + 13.5x - 4.5")
