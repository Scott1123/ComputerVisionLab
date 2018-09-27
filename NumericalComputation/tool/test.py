#!/usr/bin/python3
from tool.equations_solver import EquationsSolver
import numpy as np


def main():
    dim = 100
    A = np.zeros((dim, dim), dtype='f8')
    for i in range(dim):
        for j in range(dim):
            if i == j:
                A[i, j] = 2
            elif np.abs(i - j) == 1:
                A[i, j] = -1
    b = np.ones((dim, 1), dtype='f8')

    # compare different methods
    x1 = EquationsSolver.solve(A, b, verbose=1, method='gauss')
    x2 = EquationsSolver.solve(A, b, verbose=1, method='lu')
    x3 = EquationsSolver.solve(A, b, verbose=1, method='chase')
    x4 = EquationsSolver.solve(A, b, verbose=1, method='jacobi')
    x5 = EquationsSolver.solve(A, b, verbose=1, method='gauss_seidel')
    x6 = EquationsSolver.solve(A, b, verbose=1, method='sor')

    print('%8s %8s %8s %8s %8s %8s' % ('Gauss', 'LU', 'chase', 'Jacobi', 'GauSeid', 'SOR'))
    print('-' * 54)
    for i in range(dim):
        print('%8.2f %8.2f %8.2f %8.2f %8.2f %8.2f' % (x1[i], x2[i], x3[i], x4[i], x5[i], x6[i]))

    # Here is other data to test the top four methods

    # A = np.random.randint(0, 10, (dim, dim)) + np.eye(dim)
    # x = np.arange(1, dim + 1).reshape((dim, 1))
    # b = A.dot(x)
    # x = EquationsSolver.solve(A, b, method='Gauss')
    # x = EquationsSolver.solve(A, b, method='LU')

    # A, x, b = model.generate_data()

    # LU data 1
    # A = np.array([[2, 1, 5], [4, 1, 12], [-2, -4, 5]], dtype='f8')
    # x = np.array([[1], [-1], [2]], dtype='f8')
    # b = np.array([[11], [27], [12]], dtype='f8')

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
    #
    # x2 = EquationsSolver.solve(A, b, method='square_root')
    # print(x2)
    # print(x)


if __name__ == '__main__':
    main()
