import numpy as np
from scipy.sparse.linalg import cgs

class MatrixSolver:
    def __init__(self, Z, V):
        self.impedance_matrix = np.array(Z)
        self.forcing_vector = np.array(V)

    def solve(self, tol=1e-6, max_iter=10000):
        I, info = cgs(self.impedance_matrix, self.forcing_vector, tol=tol, maxiter=max_iter)
        if info == 0:
            # Successful convergence
            print("Conjugate Gradient Solver converged successfully.")
            return I
        elif info > 0:
            # Did not converge within max_iter iterations
            print(f"Conjugate Gradient Solver did not converge within {max_iter} iterations.")
            return I
        else:
            raise ValueError("Unsupported method")



