#!/usr/bin/python3
from solve_equations import EquationsSolver
import numpy as np


def main():
    model = EquationsSolver()
    dim = 10
    A = np.random.randint(0, 10, (dim, dim)) + np.eye(dim)
    x = np.arange(1, dim + 1).reshape((dim, 1))
    b = A.dot(x)
    model.solve(A, b, method='Gauss')
    # model.solve(A, b, method='LU')

    # Here is other data to test four methods

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


if __name__ == '__main__':
    main()
