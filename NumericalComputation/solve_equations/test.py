#!/usr/bin/python3
from homework_2.solve_equations import EquationsSolver
import numpy as np


def main():
    dim = 100
    A = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i == j:
                A[i, j] = 2
            elif np.abs(i - j) == 1:
                A[i, j] = -1
    b = np.ones((dim, 1))

    # compare the two answers (Jacobi, Gauss_Seidel and chase method)

    xj = EquationsSolver.solve(A, b, method='jacobi')
    xg = EquationsSolver.solve(A, b, method='gauss_seidel')
    xc = EquationsSolver.solve(A, b, method='chase')
    for i in range(dim):
        print('%4d: xc: %8.2f\t\t'
              '[xj: %8.2f\t loss: %8.2f]\t\t'
              '[xg: %8.2f\t loss: %8.2f]'
              % (i+1, xc[i, 0],
                 xj[i, 0], np.abs(xj[i, 0] - xc[i, 0]),
                 xg[i, 0], np.abs(xg[i, 0] - xc[i, 0])))

    # Here is other data to test the top four methods

    # A = np.random.randint(0, 10, (dim, dim)) + np.eye(dim)
    # x = np.arange(1, dim + 1).reshape((dim, 1))
    # b = A.dot(x)
    # x = EquationsSolver.solve(A, b, method='Gauss')
    # x = EquationsSolver.solve(A, b, method='LU')

    # A, x, b = model.generate_data()

    # LU data 1
    # A = np.array([[2, 1, 5], [4, 1, 12], [-2, -4, 5]])
    # x = np.array([[1], [-1], [2]])
    # b = np.array([[11], [27], [12]])

    # chase data 1
    # A = np.array([[2, 1, 0, 0], [1/2, 2, 1/2, 0], [0, 1/2, 2, 1/2], [0, 0, 1, 2]])
    # x = np.array([[-13/45], [7/90], [-1/45], [1/90]])
    # b = np.array([[-1/2], [0], [0], [0]])

    # chase data 2
    # A = np.array([[3, 1, 0, 0], [1, 4, 1, 0], [0, 2, 5, 2], [0, 0, 2, 6]])
    # x = np.array([[1], [3], [-2], [1]])
    # b = np.array([[6], [11], [-2], [2]])

    # square root data 1
    # A = np.array([[4, 2, -2], [2, 2, -3], [-2, -3, 14]])
    # x = np.array([[2], [2], [1]])
    # b = np.array([[10], [5], [4]])

    # x2 = EquationsSolver.solve(A, b, method='square_root')
    # print(x2)
    # print(x)


if __name__ == '__main__':
    main()
