# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy
import sympy


# pylint: disable=too-many-locals
def tree(n, X, mu=None, symbolic=False):
    '''Evaluates the entire tree of orthogonal polynomials on the unit disk for
    the weight function

        W_{\\mu}(x, y) = (\\mu + \\frac{1}{2}) (1-x^2-y^2)^{\\mu-1/2},

    \\mu>-1/2. The default choice is \\mu=1/2, giving W_1(x, y) = 1.

    The return value is a list of arrays, where `out[k]` hosts the `2*k+1`
    values of the `k`th level of the tree

        (0, 0)
        (0, 1)   (1, 1)
        (0, 2)   (1, 2)   (2, 2)
          ...      ...      ...

    This is based on the first orthonormal basis in section 3.2 of

    Yuan Xu,
    Orthogonal polynomials of several variables,
    <https://arxiv.org/pdf/1701.02709.pdf>,

    specifically equation (3.4)

      P_k^n(x, y) = h_{k,}^{-1}
                    C_{n-k}^{k+\\mu+1/2}(x)
                    \\sqrt{1-x^2}^k C_k^{\\mu}(y/\\sqrt(1-x^2)

    with C_n^{\\lambda} being the Gegenbauer polynomial, scaled such that
    $C_n^{\\lambda}(1)=\\frac{(\\lambda+1)_n}{n!}$ and

      h_{k,n}^2 = \\frac{(2k+2\\mu+1)_{n-k} (2\\mu)_k (\\mu)_k (\\mu+1/2)}
                        {(n-k)!k! (\\mu+1/2)_k (n+\\mu+1/2)}.

    The recurrence coefficients are retrieved by exploiting the Gegenbauer
    recurrence

       C_n^{\\lambda}(t) =
           1/n (
               + 2t (n+\\lambda+1) C_{n-1}^{\\lambda}(t)
               - (n+2\\lambda-2) C_{n-2}^{\\lambda}(t)
               )

    One gets

        P_k^n(x, y) = + 2 \\alpha_{k,n} x P_k^{n-1}(x, y)
                      - \\beta_{k, n} P_k^{n-2}(x, y)

    with

        \\alpha_{k, n}^2 = \\frac{(n+\\mu+1/2)(n+\\mu-1/2)}{(n-k)(n+k+2\\mu)},
        \\beta_{k, n}^2 = \\frac{(n-k-1) (n+\\mu+1/2)(n+k+2\\mu-1)}
                                {(n-k)(n+\\mu-3/2)(n+k+2\\mu)},

    and

        P_n^n(x, y) = + 2 \\gamma_{k,n} y P_{n-1}^{n-1}(x, y)
                      - \\delta_{k, n} (1-x^2) P_{n-1}^{n-2}(x, y)

    with

        \\gamma_{k, n}^2 = \\frac{(n+\\mu-1)(n+\\mu+1/2)}{n (n+2\\mu-1)},
        \\delta_{k, n}^2 = \\frac{(n+2\\mu-2) (n-1) (n+\\mu-1/2) (n+\\mu+1/2)}
                                 {n (n+2\\mu-1) (n+\\mu-1) (n+\\mu-2)}.
    '''
    frac = sympy.Rational if symbolic else lambda x, y: x/y
    sqrt = sympy.sqrt if symbolic else numpy.sqrt
    pi = sympy.pi if symbolic else numpy.pi

    if mu is None:
        mu = frac(1, 2) if symbolic else 0.5

    p0 = 1 / sqrt(pi)

    def alpha(n):
        return numpy.array([2 * sqrt(frac(
            (n+mu+frac(1, 2)) * (n+mu-frac(1, 2)),
            (n-k) * (n+k+2*mu)
            ))
            for k in range(n)
            ])

    def beta(n):
        return numpy.array([sqrt(frac(
            (n-1-k) * (n+mu+frac(1, 2)) * (n+k+2*mu-1),
            (n-k) * (n+mu-frac(3, 2)) * (n+k+2*mu)
            ))
            for k in range(n-1)
            ])

    def gamma(n):
        return 2 * sqrt(frac(
            (n+mu-1) * (n+mu+frac(1, 2)),
            (n+2*mu-1) * n
            ))

    def delta(n):
        return sqrt(frac(
            (n-1) * (n+2*mu-2) * (n+mu-frac(1, 2)) * (n+mu+frac(1, 2)),
            n * (n+2*mu-1) * (n+mu-1) * (n+mu-2)
            ))

    out = [numpy.array([0 * X[0] + p0])]

    one_min_x2 = 1 - X[0]**2

    for L in range(1, n+1):
        out.append(numpy.concatenate([
            out[L-1] * numpy.multiply.outer(alpha(L), X[0]),
            [out[L-1][L-1] * gamma(L) * X[1]],
            ])
            )

        if L > 1:
            out[-1][:L-1] -= (out[L-2].T * beta(L)).T
            out[-1][-1] -= out[L-2][L-2] * delta(L) * one_min_x2

    return out
