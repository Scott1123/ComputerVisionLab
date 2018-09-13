import numpy as np


class EquationsSolver(object):
    """
    This class is for solving equations in some methods, including Gauss, LU decompose, chase, square root.

    """
    __slots__ = ['n', 'eps']
    def __init__(self, eps=1e-5):
        np.random.seed(0)
        self.eps = eps

    def generate_data(self, n=10):
        """
        Generate Unrestricted Matrix A, and vector x and b.
        :param n: the scale of matrix
        :return: matrix A, x, b
        """
        A = np.random.randint(0, 10, (n, n))
        A = A + np.eye(n)
        x = np.random.randint(1, 10, (n, 1))
        b = A.dot(x)
        return A, x, b

    def generate_tridiagonal_matrix(self, n=10):
        """
        Generate Tridiagonal Matrix A, and vector x and b.
        :param n: the scale of matrix
        :return: matrix A, x, b
        """
        A = np.zeros((n, n))
        # diag_2 is the main diag
        diag_1 = np.random.randint(1, 8, n)
        diag_2 = np.random.randint(20, 30, n)
        diag_3 = np.random.randint(1, 8, n)
        A[0, 0] = diag_2[0]
        for i in range(1, n):
            A[i, i-1] = diag_1[i]
            A[i, i] = diag_2[i]
            A[i-1, i] = diag_3[i]
        x = np.random.randint(1, 10, (n, 1))
        b = A.dot(x)
        return A, x, b

    def solve(self, A, b, method='Gauss'):
        """
        Solve equations in specified method.

        :param A: coefficient matrix of the equations
        :param b: vector
        :param method: the way to solve equations
        :return: the solution x or error infomation
        """
        # self._show_equations(A, b)  # only when dim <= 10

        func = {
            'Gauss': self._solve_Gauss,
            'LU': self._solve_LU,
            'chase': self._solve_chase,
            'square_root': self._solve_square_root
        }.get(method, self._other_method)
        flag, answer = func(A, b)
        if flag == -1:
            print('This method is not supported!\n'
                  'Please choose one from these:\n'
                  '(Gauss, LU, chase, square_root).')
        elif flag == 0:
            print('No Answer! det(A) = 0.')
        elif flag == 1:
            print('Here is the solution x:')
            print(answer)
        elif flag == 2:
            print('No Answer! Matrix A is not a tridiagonal matrix.')
        elif flag == 3:
            print('No Answer! The main doagonal of Matrix A is not big enough.')
        elif flag == 4:
            print('No Answer! Matrix A is not a symmetric matrix.')
        elif flag == 5:
            print('No Answer! Matrix A is not a positive definite matrix.')

        return answer

    def _show_equations(self, A, b):
        n = len(A)
        if n <= 10:
            print('Here is the equations:')
            for i in range(n):
                lst = ['{:4.0f}'.format(i) for i in A[i, :]]
                bi = '{:6.0f}'.format(b[i, 0])
                print('[{}] [x{}] = [{}]'.format(' '.join(lst), i, bi))
            print('===========================================')

    def _solve_Gauss(self, A, b):
        # 1. update coefficient
        n = len(b)
        for k in range(n - 1):
            # find the max coefficient
            line = k
            for i in range(k + 1, n):
                if A[i, k] > A[line, k]:
                    line = i
            if np.abs(A[line, k]) < self.eps:
                return 0, None
            if line != k:
                for j in range(n):
                    A[line, j], A[k, j] = A[k, j], A[line, j]
                b[line, 0], b[k, 0] = b[k, 0], b[line, 0]

            # update(k)
            for i in range(k + 1, n):
                A[i, k] /= A[k, k]
                for j in range(k + 1, n):
                    A[i, j] -= (A[i, k] * A[k, j])
                b[i, 0] -= (A[i, k] * b[k, 0])

        if np.abs(A[n - 1, n - 1]) < self.eps:
            return 0, None

        # 2. solve Ax = b
        b[n - 1, 0] /= A[n - 1, n - 1]
        for i in range(n - 2, -1, -1):
            sum = 0
            for j in range(i + 1, n):
                sum += A[i, j] * b[j, 0]
            b[i, 0] = (b[i, 0] - sum) / A[i, i]

        return 1, b

    def _solve_LU(self, A, b):
        # Doolittle method
        # 0. judge whether A can be decomposed into L and U.
        if not self._judge_n_det(A):
            return 0, None

        n = len(b)
        # 1. get L & U (L & U are stored in matrix A)
        for i in range(1, n):
            A[i, 0] /= A[0, 0]
        for k in range(1, n):
            for i in range(k, n):
                sum = 0
                for j in range(k):
                    sum += (A[k, j] * A[j, i])
                A[k, i] -= sum
            for i in range(k+1, n):
                sum = 0
                for j in range(k):
                    sum += (A[i, j] * A[j, k])
                A[i, k] = (A[i, k] - sum) / A[k, k]

        # 2. solve Ly = b
        y = np.zeros((n, 1))
        y[0, 0] = b[0, 0]
        for k in range(1, n):
            sum = 0
            for j in range(k):
                sum += (A[k, j] * y[j, 0])
            y[k, 0] = b[k, 0] - sum

        # 3. solve Ux = y (x is stored in b)
        b[n-1, 0] = y[n-1, 0] / A[n-1, n-1]
        for k in range(n-2, -1, -1):
            sum = 0
            for j in range(k+1, n):
                sum += (A[k, j] * b[j, 0])
            b[k, 0] = (y[k, 0] - sum) / A[k, k]

        return 1, b

    def _solve_chase(self, A, f):
        # 0. judge whether A is a tridiagonal matrix
        if not self._judge_tridiagonal_matrix(A):
            return 2, None

        # 1. get diags and check main diagonal
        n = len(A)
        a, b, c = np.zeros(n), np.zeros(n), np.zeros(n) # b is the main diag
        b[0] = A[0, 0]
        for i in range(1, n):
            a[i] = A[i, i - 1]
            b[i] = A[i, i]
            c[i-1] = A[i - 1, i]
        c[n-1] = 0

        if not self._judge_main_diag(a, b, c, n):
            return 3, None

        # 2. get alpha/beta/gamma
        alpha = np.zeros(n)
        beta = np.zeros(n)
        gamma = a[0:n]
        beta[0] = c[0] / b[0]
        for i in range(1, n-1):
            beta[i] = c[i] / (b[i] - a[i] * beta[i - 1])
        alpha[0] = b[0]
        for i in range(1, n):
            alpha[i] = b[i] - a[i] * beta[i - 1]

        # 3. solve Ly = f
        y = np.zeros(n).reshape((n, 1))
        y[0, 0] = f[0, 0] / b[0]
        for i in range(1, n):
            y[i, 0] = (f[i, 0] - a[i] * y[i - 1, 0]) / (b[i] - a[i] * beta[i - 1])

        # 4. solve Ux = y (x is also stored in f)
        f[n-1, 0] = y[n-1, 0]
        for i in range(n-2, -1, -1):
            f[i, 0] = y[i] - (beta[i] * f[i + 1, 0])

        return 1, f

    def _solve_square_root(self, A, b):
        # 0. judge whether A is a symmetric and positive definite matrix
        if not self._judge_symmetric_matrix(A):
            return 4, None
        if not self._judge_positive_definite_matrix(A):
            return 5, None

        # 1. get L (stored in A)
        n = len(A)
        A[0, 0] = np.sqrt(A[0, 0])
        for i in range(1, n):
            A[i, 0] /= A[0, 0]
        for j in range(1, n):
            # calculate L[j, j] first
            sum = 0
            for k in range(0, j):
                sum += (A[j, k] * A[j, k])
            A[j, j] = np.sqrt(A[j, j] - sum)
            # calculate coefficients under L[j, j]
            for i in range(j+1, n):
                sum = 0
                for k in range(0, j):
                    sum += (A[i, k] * A[j, k])
                A[i, j] = (A[i, j] - sum) / A[j, j]

        # 2. solve Ly = b
        y = np.zeros(n).reshape((n, 1))
        y[0, 0] = b[0, 0] / A[0, 0]
        for i in range(1, n):
            sum = 0
            for k in range(0, i):
                sum += (A[i, k] * y[k, 0])
            y[i, 0] = (b[i, 0] - sum) / A[i, i]

        # 3. solve L(T)x = y (x is stored in b)
        b[n-1, 0] = y[n-1] / A[n-1, n-1]
        for i in range(n-2, -1, -1):
            sum = 0
            for k in range(i+1, n):
                sum += (A[k, i] * b[k, 0])
            b[i] = (y[i, 0] - sum) / A[i, i]

        return 1, b

    def _other_method(self, A, b):
        return -1, None

    # made for LU
    def _judge_n_det(self, A):
        n = len(A)
        for i in range(1, n+1):
            mat = A[:i, :i]
            if np.linalg.det(mat) == 0:
                return False
        return True

    # made for chase 1
    def _judge_tridiagonal_matrix(self, A):
        n = len(A)
        for i in range(n):
            for j in range(n):
                if np.abs(i - j) <= 1:
                    continue
                if A[i, j] != 0:
                    return False
        return True

    # made for chase 2
    def _judge_main_diag(self, a, b, c, n):
        if not np.abs(b[0]) > np.abs(c[0]) > 0:
            return False
        for i in range(1, n-1):
            if np.abs(b[i]) >= np.abs(a[i]) + np.abs(c[i]) and a[i] * c[i] != 0:
                continue
            else:
                return False
        if not np.abs(b[n-1]) > np.abs(a[n-1]) > 0:
            return False
        return True

    # made for square_root 1
    def _judge_symmetric_matrix(self, A):
        n = len(A)
        for i in range(n):
            for j in range(i):
                if A[i, j] != A[j, i]:
                    return False
        return True

    # made for square_root 2
    def _judge_positive_definite_matrix(self, A):
        eigvals = np.linalg.eigvals(A)
        return np.all(eigvals > 0)
