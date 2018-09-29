#!/usr/bin/python3
from tool.equations_solver import EquationsSolver
dim = 100


# test different methods
def test_methods(A, b, methods_list):
    methods_num = len(methods_list)

    # get solution
    solution_x = []
    for i in range(methods_num):
        tmp_x = EquationsSolver.solve(A, b, verbose=1, method=methods_list[i])
        solution_x.append(tmp_x)

    # make table head
    for i in range(methods_num):
        print('%12s' % methods_list[i], end=' ')
    print()
    print('-' * (13 * methods_num))

    # show solution
    for k in range(dim):
        for i in range(methods_num):
            print('%12.2f' % solution_x[i][k, 0], end=' ')
        print()


def get_my_data():
    pass
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


def main():
    A, b = EquationsSolver.generate_homework_data(dim=100)
    # methods_list = ['gauss', 'lu', 'chase', 'jacobi', 'gauss_seidel', 'sor', 'cg']
    methods_list = ['qr']
    test_methods(A, b, methods_list)


if __name__ == '__main__':
    main()
