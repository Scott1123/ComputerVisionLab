import numpy as np
from datetime import datetime as dt


class EquationsSolver(object):
    """
    This class is for solving equations in some methods, including Gauss, LU decompose, chase, square root.
    add Jacobi itration method, Gauss_Seidel itration mathod.

    """
    eps = 1e-6
    omega = 1.9375
    max_itration_times = 100000

    @classmethod
    def generate_data(cls, n=10):
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

    @classmethod
    def generate_tridiagonal_matrix(cls, n=10):
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

    @classmethod
    def solve(cls, A, b, method='gauss', eps=1e-6, max_itration_times=100000, omega=1.9375):
        """
        Solve equations in specified method.

        :param A: coefficient matrix of the equations
        :param b: vector
        :param method: the way to solve equations
        :param max_itration_times: the maximum rounds of iteration
        :return: the solution x or error information
        """
        # cls.show_equations(A, b)  # only when dim <= 10
        start = dt.now()
        cls.eps = eps
        cls.max_itration_times = max_itration_times
        cls.omega = omega
        func = {
            'gauss': cls._solve_gauss,
            'lu': cls._solve_lu,
            'chase': cls._solve_chase,
            'square_root': cls._solve_square_root,
            'jacobi': cls._solve_jacobi,
            'gauss_seidel': cls._solve_gauss_seidel,
            'sor': cls._solve_sor
        }.get(method, cls._other_method)
        # make a copy of A and b to make sure they will not be changed.
        A0 = np.copy(A)
        b0 = np.copy(b)
        flag, answer = func(A0, b0)
        if flag == -1:
            print('This method is not supported!\n'
                  'Please choose one from these:\n'
                  '(gauss, lu, chase, square_root, jacobi, gauss_seidel).')
        elif flag == 0:
            print('No Answer! det(A) = 0.')
        elif flag == 1:
            print('[%s] Success!' % method)
        elif flag == 2:
            print('No Answer! Matrix A is not a tridiagonal matrix.')
        elif flag == 3:
            print('No Answer! The main diagonal of Matrix A is not big enough.')
        elif flag == 4:
            print('No Answer! Matrix A is not a symmetric matrix.')
        elif flag == 5:
            print('No Answer! Matrix A is not a positive definite matrix.')
        elif flag == 6:
            print('No Answer! A[i, i] == 0 for some i.')
        print('[%s] time cost: %.4f s.' % (method, (dt.now() - start).total_seconds()))
        return answer

    @classmethod
    def show_equations(cls, A, b):
        n = len(A)
        if n <= 10:
            print('Here is the equations:')
            for i in range(n):
                Ax_i = ['{:4.0f}'.format(i) for i in A[i, :]]
                b_i = '{:6.0f}'.format(b[i, 0])
                print('[{}] [x{}] = [{}]'.format(' '.join(Ax_i), i, b_i))
            print('===========================================')

    @classmethod
    def _solve_gauss(cls, A, b):
        n = len(A)
        # 1. update coefficient
        for k in range(n - 1):
            # find the max coefficient
            line = k
            for i in range(k + 1, n):
                if A[i, k] > A[line, k]:
                    line = i
            if np.abs(A[line, k]) < cls.eps:
                return 0, None
            if line != k:
                for j in range(n):
                    A[line, j], A[k, j] = A[k, j], A[line, j]
                b[line, 0], b[k, 0] = b[k, 0], b[line, 0]

            # update(k)
            for i in range(k + 1, n):
                A[i, k] = A[i, k] / A[k, k]
                # print('---- ', A[i, k])
                for j in range(k + 1, n):
                    A[i, j] = A[i, j] - (A[i, k] * A[k, j])
                b[i, 0] = b[i, 0] - (A[i, k] * b[k, 0])

        if np.abs(A[n - 1, n - 1]) < cls.eps:
            return 0, None

        # 2. solve Ax = b
        b[n - 1, 0] /= A[n - 1, n - 1]
        for i in range(n - 2, -1, -1):
            tmp_sum = 0
            for j in range(i + 1, n):
                tmp_sum += A[i, j] * b[j, 0]
            b[i, 0] = (b[i, 0] - tmp_sum) / A[i, i]

        return 1, b

    @classmethod
    def _solve_lu(cls, A, b):
        # Doolittle method
        n = len(A)

        # 0. judge whether A can be decomposed into L and U.
        if not cls._judge_n_det(A):
            return 0, None

        # 1. get L & U (L & U are stored in matrix A)
        for i in range(1, n):
            A[i, 0] /= A[0, 0]
        for k in range(1, n):
            for i in range(k, n):
                tmp_sum = 0
                for j in range(k):
                    tmp_sum += (A[k, j] * A[j, i])
                A[k, i] -= tmp_sum
            for i in range(k+1, n):
                tmp_sum = 0
                for j in range(k):
                    tmp_sum += (A[i, j] * A[j, k])
                A[i, k] = (A[i, k] - tmp_sum) / A[k, k]

        # 2. solve Ly = b
        y = np.zeros((n, 1))
        y[0, 0] = b[0, 0]
        for k in range(1, n):
            tmp_sum = 0
            for j in range(k):
                tmp_sum += (A[k, j] * y[j, 0])
            y[k, 0] = b[k, 0] - tmp_sum

        # 3. solve Ux = y
        x = np.zeros((n, 1))
        x[n-1, 0] = y[n-1, 0] / A[n-1, n-1]
        for k in range(n-2, -1, -1):
            tmp_sum = 0
            for j in range(k+1, n):
                tmp_sum += (A[k, j] * x[j, 0])
            x[k, 0] = (y[k, 0] - tmp_sum) / A[k, k]

        return 1, x

    @classmethod
    def _solve_chase(cls, A, f):
        # 0. judge whether A is a tridiagonal matrix
        if not cls._judge_tridiagonal_matrix(A):
            return 2, None

        # 1. get diags and check main diagonal
        n = len(A)
        a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)  # b is the main diag
        b[0] = A[0, 0]
        for i in range(1, n):
            a[i] = A[i, i - 1]
            b[i] = A[i, i]
            c[i-1] = A[i - 1, i]
        c[n-1] = 0

        if not cls._judge_main_diag(a, b, c, n):
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

        # 4. solve Ux = y
        x = np.zeros((n, 1))
        x[n-1, 0] = y[n-1, 0]
        for i in range(n-2, -1, -1):
            x[i, 0] = y[i] - (beta[i] * x[i + 1, 0])

        return 1, x

    @classmethod
    def _solve_square_root(cls, A, b):
        n = len(A)

        # 0. judge whether A is a symmetric and positive definite matrix
        if not cls._judge_symmetric_matrix(A):
            return 4, None
        if not cls._judge_positive_definite_matrix(A):
            return 5, None

        # 1. get L
        L = np.zeros((n, n))
        L[0, 0] = np.sqrt(A[0, 0])
        for i in range(1, n):
            L[i, 0] = A[i, 0] / L[0, 0]
        for j in range(1, n):
            # calculate L[j, j] first
            tmp_sum = 0
            for k in range(0, j):
                tmp_sum += (L[j, k] * L[j, k])
            L[j, j] = np.sqrt(A[j, j] - tmp_sum)
            # calculate coefficients under L[j, j]
            for i in range(j+1, n):
                tmp_sum = 0
                for k in range(0, j):
                    tmp_sum += (L[i, k] * L[j, k])
                L[i, j] = (A[i, j] - tmp_sum) / L[j, j]

        # 2. solve Ly = b
        y = np.zeros((n, 1))
        y[0, 0] = b[0, 0] / L[0, 0]
        for i in range(1, n):
            tmp_sum = 0
            for k in range(0, i):
                tmp_sum += (L[i, k] * y[k, 0])
            y[i, 0] = (b[i, 0] - tmp_sum) / L[i, i]

        # 3. solve L(T)x = y
        x = np.zeros((n, 1))
        x[n-1, 0] = y[n-1] / L[n-1, n-1]
        for i in range(n-2, -1, -1):
            tmp_sum = 0
            for k in range(i+1, n):
                tmp_sum += (L[k, i] * x[k, 0])
            x[i] = (y[i, 0] - tmp_sum) / L[i, i]

        return 1, x

    @classmethod
    def _solve_jacobi(cls, A, b):
        n = len(A)

        # 1. get Dc, L+U, Bj, fj
        D_inv = np.zeros((n, n))
        LplusU = -A[:, :]
        for i in range(n):
            if A[i, i] == 0:
                return 6, None
            D_inv[i, i] = 1 / A[i, i]
            LplusU[i, i] = 0
        Bj = D_inv.dot(LplusU)
        fj = D_inv.dot(b)

        # 2. x[k+1] = Bj(x[k]) + fj
        times = 0
        x = np.zeros((n, 1))
        b2 = A.dot(x)
        while times < cls.max_itration_times and not cls._judge_convergence(b2, b):
            x = Bj.dot(x) + fj
            b2 = A.dot(x)
            times += 1
        print('[jacobi] itration times:', times)
        return 1, x

    @classmethod
    def _solve_gauss_seidel(cls, A, b):
        n = len(A)

        # 1. get D, L, U, Bg, fg
        DminusL = np.zeros((n, n))
        U = np.zeros((n, n))
        for i in range(n):
            DminusL[i, :(i+1)] = A[i, :(i+1)]
            U[i, (i+1):] = -A[i, (i+1):]
        DminusL_inv = np.linalg.inv(DminusL)
        Bg = DminusL_inv.dot(U)
        fg = DminusL_inv.dot(b)

        # 2. x[k+1] = Bg(x[k]) + fg
        times = 0
        x = np.zeros((n, 1))
        b2 = A.dot(x)
        while times < cls.max_itration_times and not cls._judge_convergence(b2, b):
            x = Bg.dot(x) + fg
            b2 = A.dot(x)
            times += 1
        print('[gauss_seidel] itration times:', times)
        return 1, x

    @classmethod
    def _solve_sor(cls, A, b):
        n = len(A)

        # 1. get Bw, fw
        D = np.zeros((n, n))
        Dminus_wL = np.zeros((n, n))
        U = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    D[i, j] = A[i, j]
                    Dminus_wL[i, j] = A[i, j]
                elif i < j:
                    U[i, j] = -A[i, j]
                else:
                    Dminus_wL[i, j] = cls.omega * A[i, j]
        Dminus_wL_inv = np.linalg.inv(Dminus_wL)
        Bw = Dminus_wL_inv.dot((1 - cls.omega) * D + cls.omega * U)
        fw = cls.omega * Dminus_wL_inv.dot(b)

        # 2. x[k+1] = Bw(x[k]) + fw
        times = 0
        x = np.zeros((n, 1))
        b2 = A.dot(x)
        while times < cls.max_itration_times and not cls._judge_convergence(b2, b):
            x = Bw.dot(x) + fw
            b2 = A.dot(x)
            times += 1
        print('[sor] itration times:', times)
        return 1, x

    @classmethod
    def _other_method(cls, A, b):
        return -1, None

    # made for LU
    @classmethod
    def _judge_n_det(cls, A):
        n = len(A)
        for i in range(1, n+1):
            mat = A[:i, :i]
            if np.linalg.det(mat) == 0:
                return False
        return True

    # made for chase 1
    @classmethod
    def _judge_tridiagonal_matrix(cls, A):
        n = len(A)
        for i in range(n):
            for j in range(n):
                if np.abs(i - j) <= 1:
                    continue
                if A[i, j] != 0:
                    return False
        return True

    # made for chase 2
    @classmethod
    def _judge_main_diag(cls, a, b, c, n):
        return np.all((np.abs(b) - np.abs(a) - np.abs(c)) >= 0)

    # made for square_root 1
    @classmethod
    def _judge_symmetric_matrix(cls, A):
        return np.all(A == A.T)

    # made for square_root 2
    @classmethod
    def _judge_positive_definite_matrix(cls, A):
        eig_vals = np.linalg.eigvals(A)
        return np.all(eig_vals > 0)

    # made for jacobi and gauss_seidel
    @classmethod
    def _judge_convergence(cls, b2, b):
        return np.all(np.abs(b2 - b) < cls.eps)
