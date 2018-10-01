#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time  : 2018/9/18 19:17
# @Author: Scott Yang
# @File  : test.py

"""

This is an equations solver.

Provides some classic methods to solve equations, including Gauss, LU decompose, chase, square root.
add Jacobi itration method, Gauss_Seidel itration mathod.
add SOR method.
add Conjugate Gradient Method.
add QR decomposition method.

"""

import numpy as np
from datetime import datetime as dt


_verbose = 0
_eps = 1e-6
_omega = 1.9375
_max_itration_times = 100000


class EquaSolError(Exception):
    """
    Generic Python-exception-derived object raised by linalg functions.

    General purpose exception class, derived from Python's exception.Exception
    class, programmatically raised in linalg functions when a Linear
    Algebra-related condition would prevent further correct execution of the
    function.
    (from numpy.linalg)
    """
    pass


def generate_homework_data(n=100):
    """
    Generate homework data.
    :param n: the scale of matrix
    :return: matrix A, b
    """
    A = np.zeros((n, n), dtype='f8')
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 2
            elif np.abs(i - j) == 1:
                A[i, j] = -1
    b = np.ones((n, 1), dtype='f8')
    return A, b

 
def generate_random_data(n=10):
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

 
def generate_tridiagonal_matrix(n=10):
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


def _raise_equasolerror_det_0(method):
    raise EquaSolError('[%s] det(A) = 0.' % method)


def _raise_equasolerror_not_tridiagonal(method):
    raise EquaSolError('[%s] Matrix A is not a tridiagonal matrix.' % method)


def _raise_equasolerror_not_main_diagonal(method):
    raise EquaSolError('[%s] The main diagonal of Matrix A is not big enough.' % method)


def _raise_equasolerror_not_symmetric(method):
    raise EquaSolError('[%s] Matrix A is not a symmetric matrix.' % method)


def _raise_equasolerror_not_positive_definite(method):
    raise EquaSolError('[%s] Matrix A is not a positive definite matrix.' % method)


def _raise_equasolerror_0_in_main_diagonal(method):
    raise EquaSolError('[%s] A[i, i] == 0 for some i.' % method)


def _raise_equasolerror_omega_out_of_range(method):
    raise EquaSolError('[%s] Out of range. Omega should be in range(0, 2).' % method)


def _raise_equasolerror_no_method(method):
    raise EquaSolError('[%s] This method is not supported!'
                       '(gauss, lu, chase, square_root, jacobi, gauss_seidel, sor, cg, qr).'
                       % method)

 
def solve(A, b, method='gauss', verbose=0, eps=1e-6, max_itration_times=100000, omega=1.9375):
    """
    Solve equations in specified method.

    :param A: coefficient matrix of the equations
    :param b: vector
    :param method: the way to solve equations
    :param verbose: whether show the running information
    :param eps: *epsilon*
    :param max_itration_times: the maximum *rounds* of iteration
    :param omega: *relaxation factor* for SOR method.
    :return: the solution x or 'None' if error occurs
    """
    # cls.show_equations(A, b)  # only when dim <= 10
    start = dt.now()
    global _verbose, _eps, _max_itration_times, _omega
    _verbose = verbose
    _eps = eps
    _max_itration_times = max_itration_times
    _omega = omega
    func = {
        'gauss': _solve_gauss,
        'lu': _solve_lu,
        'chase': _solve_chase,
        'square_root': _solve_square_root,
        'jacobi': _solve_jacobi,
        'gauss_seidel': _solve_gauss_seidel,
        'sor': _solve_sor,
        'cg': _solve_cg,
        'qr': _solve_qr
    }.get(method, 'other_method')
    if func == 'other_method':
        _raise_equasolerror_no_method(method)
    # make a copy of A and b to make sure they will not be changed.
    # show_equations(A, b)
    A0 = np.copy(A)
    b0 = np.copy(b)
    answer = func(A0, b0)
    if _verbose == 1:
        print('[%s] time cost: %.4f s.' % (method, (dt.now() - start).total_seconds()))

    return answer

 
def show_equations(A, b):
    n = len(A)
    if n <= 10:
        print('Here is the equations:')
        for i in range(n):
            Ax_i = ['{:4.0f}'.format(i) for i in A[i, :]]
            b_i = '{:6.0f}'.format(b[i, 0])
            print('[{}] [x{}] = [{}]'.format(' '.join(Ax_i), i, b_i))
        print('===========================================')

 
def _solve_gauss(A, b):
    n = len(A)
    # 1. update coefficient
    for k in range(n - 1):
        # find the max coefficient
        line = k
        for i in range(k + 1, n):
            if A[i, k] > A[line, k]:
                line = i
        if np.abs(A[line, k]) < _eps:
            _raise_equasolerror_det_0('gauss')
        if line != k:
            for j in range(n):
                A[line, j], A[k, j] = A[k, j], A[line, j]
            b[line, 0], b[k, 0] = b[k, 0], b[line, 0]

        # update(k)
        for i in range(k + 1, n):
            A[i, k] = A[i, k] / A[k, k]
            for j in range(k + 1, n):
                A[i, j] = A[i, j] - (A[i, k] * A[k, j])
            b[i, 0] = b[i, 0] - (A[i, k] * b[k, 0])

    if np.abs(A[n - 1, n - 1]) < _eps:
        _raise_equasolerror_det_0('gauss')

    # 2. solve Ax = b (x is stored in b)
    b[n - 1, 0] /= A[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        tmp_sum = 0
        for j in range(i + 1, n):
            tmp_sum += A[i, j] * b[j, 0]
        b[i, 0] = (b[i, 0] - tmp_sum) / A[i, i]

    return b

 
def _solve_lu(A, b):
    # Doolittle method
    n = len(A)

    # 0. judge whether A can be decomposed into L and U.
    if not _judge_n_det(A):
        _raise_equasolerror_det_0('lu')

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

    # 3. solve Ux = y (x is stored in b)
    b[n-1, 0] = y[n-1, 0] / A[n-1, n-1]
    for k in range(n-2, -1, -1):
        tmp_sum = 0
        for j in range(k+1, n):
            tmp_sum += (A[k, j] * b[j, 0])
        b[k, 0] = (y[k, 0] - tmp_sum) / A[k, k]
    return b

 
def _solve_chase(A, f):
    n = len(A)

    # 0. judge whether A is a tridiagonal matrix
    if not _judge_tridiagonal_matrix(A):
        _raise_equasolerror_not_tridiagonal('chase')

    # 1. get diags and check main diagonal
    a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)  # b is the main diag
    b[0] = A[0, 0]
    for i in range(1, n):
        a[i] = A[i, i - 1]
        b[i] = A[i, i]
        c[i-1] = A[i - 1, i]
    c[n-1] = 0

    if not _judge_main_diag(a, b, c, n):
        _raise_equasolerror_not_main_diagonal('chase')

    # 2. get alpha/beta/gamma
    alpha = np.zeros(n)
    beta = np.zeros(n)
    gamma = a.copy()  # useless gamma
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

    # 4. solve Ux = y (x is stored in f)
    f[n-1, 0] = y[n-1, 0]
    for i in range(n-2, -1, -1):
        f[i, 0] = y[i] - (beta[i] * f[i + 1, 0])
    return f

 
def _solve_square_root(A, b):
    n = len(A)

    # 0. judge whether A is a symmetric and positive definite matrix
    if not _judge_symmetric_matrix(A):
        _raise_equasolerror_not_symmetric('square_root')
    if not _judge_positive_definite_matrix(A):
        _raise_equasolerror_not_positive_definite('square_root')

    # 1. get L (L is stored in A to save space)
    A[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, n):
        A[i, 0] = A[i, 0] / A[0, 0]
    for j in range(1, n):
        # calculate A[j, j] first
        tmp_sum = 0
        for k in range(0, j):
            tmp_sum += (A[j, k] * A[j, k])
        A[j, j] = np.sqrt(A[j, j] - tmp_sum)
        # calculate coefficients under A[j, j]
        for i in range(j + 1, n):
            tmp_sum = 0
            for k in range(0, j):
                tmp_sum += (A[i, k] * A[j, k])
            A[i, j] = (A[i, j] - tmp_sum) / A[j, j]

    # 2. solve Ly = b
    y = np.zeros((n, 1))
    y[0, 0] = b[0, 0] / A[0, 0]
    for i in range(1, n):
        tmp_sum = 0
        for k in range(0, i):
            tmp_sum += (A[i, k] * y[k, 0])
        y[i, 0] = (b[i, 0] - tmp_sum) / A[i, i]

    # 3. solve L(T)x = y (x is stored in b)
    b[n - 1, 0] = y[n - 1] / A[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        tmp_sum = 0
        for k in range(i + 1, n):
            tmp_sum += (A[k, i] * b[k, 0])
        b[i] = (y[i, 0] - tmp_sum) / A[i, i]
    return b

 
def _solve_jacobi(A, b):
    n = len(A)

    # 1. judge main diagonal and get Dc, L+U, Bj, fj
    D_inv = np.zeros((n, n))
    LplusU = -A[:, :]
    for i in range(n):
        if A[i, i] == 0:
            _raise_equasolerror_0_in_main_diagonal('jacobi')
        D_inv[i, i] = 1 / A[i, i]
        LplusU[i, i] = 0
    Bj = D_inv.dot(LplusU)
    fj = D_inv.dot(b)

    # 2. x[k+1] = Bj(x[k]) + fj
    times = 0
    x = np.zeros((n, 1))
    b2 = A.dot(x)
    while times < _max_itration_times and not _judge_convergence(b2, b):
        x = Bj.dot(x) + fj
        b2 = A.dot(x)
        times += 1
    if _verbose:
        print('[jacobi] itration times:', times)
    return x

 
def _solve_gauss_seidel(A, b):
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
    while times < _max_itration_times and not _judge_convergence(b2, b):
        x = Bg.dot(x) + fg
        b2 = A.dot(x)
        times += 1
    if _verbose:
        print('[gauss_seidel] itration times:', times)

    return x

 
def _solve_sor(A, b):
    n = len(A)

    # 0. judge whether A is a symmetric and positive definite matrix
    #    judge whether 0 < omega < 2
    if not _judge_symmetric_matrix(A):
        _raise_equasolerror_not_symmetric('sor')
    if not _judge_positive_definite_matrix(A):
        _raise_equasolerror_not_positive_definite('sor')
    if _omega <= 0 or _omega >= 2:
        _raise_equasolerror_omega_out_of_range('sor')

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
                Dminus_wL[i, j] = _omega * A[i, j]
    Dminus_wL_inv = np.linalg.inv(Dminus_wL)
    Bw = Dminus_wL_inv.dot((1 - _omega) * D + _omega * U)
    fw = _omega * Dminus_wL_inv.dot(b)

    # 2. x[k+1] = Bw(x[k]) + fw
    times = 0
    x = np.zeros((n, 1))
    b2 = A.dot(x)
    while times < _max_itration_times and not _judge_convergence(b2, b):
        x = Bw.dot(x) + fw
        b2 = A.dot(x)
        times += 1
    if _verbose:
        print('[sor] itration times:', times)

    return x

 
def _solve_cg(A, b):
    n = len(A)
    x = np.zeros((n, 1))
    r = b - A.dot(x)
    p = r.copy()

    for k in range(1, n):
        tmp_Ap = A.dot(p)
        tmp1 = np.vdot(p, tmp_Ap)
        if np.abs(tmp1) < _eps:
            break
        _a = np.vdot(r, r) / tmp1
        x = x + _a * p
        r = r - _a * tmp_Ap
        tmp2 = np.vdot(p, tmp_Ap)
        if np.abs(tmp2) < _eps:
            break
        _b = - np.vdot(r, tmp_Ap) / tmp2
        p = r + _b * p

    return x

 
def _solve_qr(A, b):
    n, m = A.shape

    # 1. get Q(here Q is Q.T), R
    Q = np.zeros((m, n))
    Q[0] = A[:, 0] / np.linalg.norm(A[:, 0], ord=2)
    for j in range(1, m):
        tmp_sum = np.zeros(n)
        for i in range(j):
            tmp_sum = tmp_sum + np.vdot(A[:, j], Q[i]) * Q[i]
        Q[j] = A[:, j] - tmp_sum
        Q[j] = Q[j] / np.linalg.norm(Q[j], ord=2)
    R = np.dot(Q, A)
    f = np.dot(Q, b)

    # 2. solve Rx = Q(T)b = f (x is stored in b)
    b[m - 1, 0] = f[m - 1, 0] / R[m - 1, m - 1]
    for k in range(m - 2, -1, -1):
        tmp_sum = 0
        for j in range(k + 1, m):
            tmp_sum += (R[k, j] * b[j, 0])
        b[k, 0] = (f[k, 0] - tmp_sum) / R[k, k]

    return b


def _judge_n_det(A):
    n = len(A)
    for i in range(1, n+1):
        mat = A[:i, :i]
        if np.linalg.det(mat) == 0:
            return False
    return True


def _judge_tridiagonal_matrix(A):
    n = len(A)
    for i in range(n):
        for j in range(n):
            if np.abs(i - j) <= 1:
                continue
            if A[i, j] != 0:
                return False
    return True


def _judge_main_diag(a, b, c, n):
    return np.all((np.abs(b) - np.abs(a) - np.abs(c)) >= 0)


def _judge_symmetric_matrix(A):
    return np.all(A == A.T)


def _judge_positive_definite_matrix(A):
    eig_vals = np.linalg.eigvals(A)
    return np.all(eig_vals > 0)


def _judge_convergence(b2, b):
    return np.linalg.norm(b2 - b, ord=2) / np.linalg.norm(b, ord=2) < _eps
