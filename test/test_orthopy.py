# -*- coding: utf-8 -*-
#
from distutils.version import LooseVersion

import math
import numpy
import orthopy
import pytest
import scipy
from scipy.special import legendre


def test_golub_welsch(tol=1.0e-14):
    '''Test the custom Gauss generator with the weight function x**2.
    '''
    alpha = 2.0

    # Get the moment corresponding to the weight function omega(x) =
    # x^alpha:
    #
    #                                     / 0 if k is odd,
    #    int_{-1}^{+1} |x^alpha| x^k dx ={
    #                                     \ 2/(alpha+k+1) if k is even.
    #
    n = 5
    k = numpy.arange(2*n+1)
    moments = (1.0 + (-1.0)**k) / (k + alpha + 1)
    alpha, beta = orthopy.golub_welsch(moments)

    assert numpy.all(abs(alpha) < tol)
    assert abs(beta[0] - 2.0/3.0) < tol
    assert abs(beta[1] - 3.0/5.0) < tol
    assert abs(beta[2] - 4.0/35.0) < tol
    assert abs(beta[3] - 25.0/63.0) < tol
    assert abs(beta[4] - 16.0/99.0) < tol

    orthopy.check_coefficients(moments, alpha, beta)
    return


def test_chebyshev(tol=1.0e-14):
    alpha = 2.0

    # Get the moment corresponding to the weight function omega(x) =
    # x^alpha:
    #
    #                                     / 0 if k is odd,
    #    int_{-1}^{+1} |x^alpha| x^k dx ={
    #                                     \ 2/(alpha+k+1) if k is even.
    #
    n = 5
    k = numpy.arange(2*n)
    moments = (1.0 + (-1.0)**k) / (k + alpha + 1)
    alpha, beta = orthopy.chebyshev(moments)

    assert numpy.all(abs(alpha) < tol)
    assert abs(beta[0] - 2.0/3.0) < tol
    assert abs(beta[1] - 3.0/5.0) < tol
    assert abs(beta[2] - 4.0/35.0) < tol
    assert abs(beta[3] - 25.0/63.0) < tol
    assert abs(beta[4] - 16.0/99.0) < tol
    return


def test_chebyshev_modified(tol=1.0e-14):
    alpha = 2.0

    # Get the moments corresponding to the Legendre polynomials and the weight
    # function omega(x) = |x^alpha|:
    #
    #                                        / 2/3   if k == 0,
    #    int_{-1}^{+1} |x^alpha| P_k(x) dx ={  8/45  if k == 2,
    #                                        \ 0     otherwise.
    #
    n = 5
    moments = numpy.zeros(2*n)
    moments[0] = 2.0/3.0
    moments[2] = 8.0/45.0
    a, b = orthopy.jacobi_recursion_coefficients(2*n, 0.0, 0.0)

    alpha, beta = orthopy.chebyshev_modified(moments, a, b)

    assert numpy.all(abs(alpha) < tol)
    assert abs(beta[0] - 2.0/3.0) < tol
    assert abs(beta[1] - 3.0/5.0) < tol
    assert abs(beta[2] - 4.0/35.0) < tol
    assert abs(beta[3] - 25.0/63.0) < tol
    assert abs(beta[4] - 16.0/99.0) < tol
    return


def test_jacobi(tol=1.0e-14):
    n = 5
    a = 1.0
    b = 1.0
    alpha, beta = orthopy.jacobi_recursion_coefficients(n, a, b)

    assert numpy.all(abs(alpha) < tol)
    assert abs(beta[0] - 4.0/3.0) < tol
    assert abs(beta[1] - 1.0/5.0) < tol
    assert abs(beta[2] - 8.0/35.0) < tol
    assert abs(beta[3] - 5.0/21.0) < tol
    assert abs(beta[4] - 8.0/33.0) < tol
    return


def test_gauss(tol=1.0e-14):
    n = 5
    points, weights = orthopy.gauss_from_coefficients(
            *orthopy.jacobi_recursion_coefficients(n, 0.0, 0.0)
            )

    s = math.sqrt(5.0 + 2*math.sqrt(10.0/7.0)) / 3.0
    t = math.sqrt(5.0 - 2*math.sqrt(10.0/7.0)) / 3.0
    assert abs(points[0] + s) < tol
    assert abs(points[1] + t) < tol
    assert abs(points[2] + 0.0) < tol
    assert abs(points[3] - t) < tol
    assert abs(points[4] - s) < tol

    u = 128.0/225.0
    v = (322.0 + 13 * math.sqrt(70)) / 900.0
    w = (322.0 - 13 * math.sqrt(70)) / 900.0
    assert abs(weights[0] - w) < tol
    assert abs(weights[1] - v) < tol
    assert abs(weights[2] - u) < tol
    assert abs(weights[3] - v) < tol
    assert abs(weights[4] - w) < tol
    return


@pytest.mark.skipif(
    LooseVersion(scipy.__version__) < LooseVersion('1.0.0'),
    reason='Requires SciPy 1.0'
    )
def test_jacobi_reconstruction(tol=1.0e-14):
    alpha1, beta1 = orthopy.jacobi_recursion_coefficients(4, 2.0, 1.0)
    points, weights = orthopy.gauss_from_coefficients(alpha1, beta1)

    alpha2, beta2 = orthopy.coefficients_from_gauss(points, weights)

    assert numpy.all(abs(alpha1 - alpha2) < tol)
    assert numpy.all(abs(beta1 - beta2) < tol)
    return


def test_eval(tol=1.0e-14):
    n = 5
    alpha, beta = orthopy.jacobi_recursion_coefficients(n, 0.0, 0.0)
    t = 1.0
    value = orthopy.evaluate_orthogonal_polynomial(alpha, beta, t)

    # Evaluating the Legendre polynomial in this way is rather unstable, so
    # don't go too far with n.
    ref = numpy.polyval(legendre(n, monic=True), t)

    assert abs(value - ref) < tol
    return


def test_clenshaw(tol=1.0e-14):
    n = 5
    alpha, beta = orthopy.jacobi_recursion_coefficients(n, 0.0, 0.0)
    t = 1.0

    a = numpy.ones(n+1)
    value = orthopy.clenshaw(a, alpha, beta, t)

    ref = math.fsum([
            numpy.polyval(legendre(i, monic=True), t)
            for i in range(n+1)])

    assert abs(value - ref) < tol
    return


@pytest.mark.skipif(
    LooseVersion(scipy.__version__) < LooseVersion('1.0.0'),
    reason='Requires SciPy 1.0'
    )
def test_gautschi_how_to_and_how_not_to():
    '''Test Gautschi's famous example from

    W. Gautschi,
    How and how not to check Gaussian quadrature formulae,
    BIT Numerical Mathematics,
    June 1983, Volume 23, Issue 2, pp 209–216,
    <https://doi.org/10.1007/BF02218441>.
    '''
    points = numpy.array([
        1.457697817613696e-02,
        8.102669876765460e-02,
        2.081434595902250e-01,
        3.944841255669402e-01,
        6.315647839882239e-01,
        9.076033998613676e-01,
        1.210676808760832,
        1.530983977242980,
        1.861844587312434,
        2.199712165681546,
        2.543839804028289,
        2.896173043105410,
        3.262066731177372,
        3.653371887506584,
        4.102376773975577,
        ])
    weights = numpy.array([
        3.805398607861561e-2,
        9.622028412880550e-2,
        1.572176160500219e-1,
        2.091895332583340e-1,
        2.377990401332924e-1,
        2.271382574940649e-1,
        1.732845807252921e-1,
        9.869554247686019e-2,
        3.893631493517167e-2,
        9.812496327697071e-3,
        1.439191418328875e-3,
        1.088910025516801e-4,
        3.546866719463253e-6,
        3.590718819809800e-8,
        5.112611678291437e-11,
        ])

    # weight function exp(-t**3/3)
    n = len(points)
    moments = numpy.array([
        3.0**((k-2)/3.0) * math.gamma((k+1) / 3.0)
        for k in range(2*n)
        ])

    alpha, beta = orthopy.coefficients_from_gauss(points, weights)
    # alpha, beta = orthopy.chebyshev(moments)

    errors_alpha, errors_beta = \
        orthopy.check_coefficients(moments, alpha, beta)

    assert numpy.max(errors_alpha) > 1.0e-2
    assert numpy.max(errors_beta) > 1.0e-2
    return


def test_show():
    import matplotlib.pyplot as plt

    for n in range(6):
        moments = numpy.zeros(2*n)
        # pylint: disable=len-as-condition
        if len(moments) > 0:
            moments[0] = 2.0/3.0
        if len(moments) > 2:
            moments[2] = 8.0/45.0
        a, b = orthopy.jacobi_recursion_coefficients(2*n, 0.0, 0.0)
        alpha, beta = orthopy.chebyshev_modified(moments, a, b)
        orthopy.plot(alpha, beta, -1.0, +1.0, normalized=True)

    plt.xlim(-1, +1)
    plt.ylim(-2, +2)
    plt.grid()
    plt.show()
    return


if __name__ == '__main__':
    test_show()
