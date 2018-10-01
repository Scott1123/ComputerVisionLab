# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time  : 2018/9/18 19:17
# @Author: Scott Yang
# @File  : test.py

import equations_solver
n = 100  # the scale of the Matrix


# test different methods
def test_methods(A, b, methods_list):
    methods_num = len(methods_list)

    # get solution
    solution_x = []
    for i in range(methods_num):
        tmp_x = equations_solver.solve(A, b, verbose=1, method=methods_list[i])
        solution_x.append(tmp_x)

    # make table head
    for i in range(methods_num):
        print('%12s' % methods_list[i], end=' ')
    print()
    print('-' * (13 * methods_num))

    # show solution
    for k in range(n):
        for i in range(methods_num):
            print('%12.2f' % solution_x[i][k, 0], end=' ')
        print()


def main():
    A, b = equations_solver.generate_homework_data()
    # 1. test single method.
    # x = equations_solver.gauss(A, b)
    # or
    # x = equations_solver.solve(A, b, 'gauss')

    # 2. compare different methods.
    # methods_list_all = ['gauss', 'lu', 'chase', 'square_root',
    #                     'jacobi', 'gauss_seidel', 'sor',
    #                     'cg', 'qr']
    methods_list = [
        'jacobi',
        'gauss_seidel',
        'sor',
        'cg',
        'qr'
        ]
    test_methods(A, b, methods_list)


if __name__ == '__main__':
    main()


# Here is other data to test the top four methods

# A = np.random.randint(0, 10, (n, n)) + np.eye(n)
# x = np.arange(1, n + 1).reshape((n, 1))
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
