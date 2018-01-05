# -*- coding: utf-8 -*-
#
from __future__ import division, print_function

import numpy
import pytest
import sympy

import orthopy


@pytest.mark.parametrize(
    'mu', [
        sympy.Rational(1, 2),
        sympy.Rational(3, 2),
        ])
def test_integral0(mu, n=4):
    '''Make sure that the polynomials are orthonormal
    '''
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    vals = numpy.concatenate(
        orthopy.disk.tree(n, numpy.array([x, y]), mu=mu, symbolic=True)
        )

    # Cartesian integration in sympy is bugged, cf.
    # <https://github.com/sympy/sympy/issues/13816>.
    # Simply transform to polar coordinates for now.
    r = sympy.Symbol('r')
    phi = sympy.Symbol('phi')

    one_half = sympy.Rational(1, 2)
    W = (mu + one_half) * (1-r**2)**(mu-one_half)

    assert sympy.integrate(
        W * r * vals[0].subs([(x, r*sympy.cos(phi)), (y, r*sympy.sin(phi))]),
        (r, 0, 1), (phi, 0, 2*sympy.pi)
        ) == sympy.sqrt(sympy.pi)

    for val in vals[1:]:
        assert sympy.integrate(
            W * r * val.subs([(x, r*sympy.cos(phi)), (y, r*sympy.sin(phi))]),
            (r, 0, 1), (phi, 0, 2*sympy.pi)
            ) == 0

    return


@pytest.mark.parametrize(
    'mu', [
        sympy.Rational(1, 2),
        sympy.Rational(3, 2),
        ])
def test_orthogonality(mu, n=3):
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    tree = numpy.concatenate(
        orthopy.disk.tree(n, numpy.array([x, y]), mu=mu, symbolic=True)
        )
    vals = tree * numpy.roll(tree, 1, axis=0)

    r = sympy.Symbol('r')
    phi = sympy.Symbol('phi')

    one_half = sympy.Rational(1, 2)
    W = (mu + one_half) * (1-r**2)**(mu-one_half)
    for val in vals:
        assert sympy.integrate(
            W * r * val.subs([(x, r*sympy.cos(phi)), (y, r*sympy.sin(phi))]),
            (r, 0, 1), (phi, 0, 2*sympy.pi)
            ) == 0
    return


@pytest.mark.parametrize(
    'mu', [
        sympy.Rational(1, 2),
        sympy.Rational(3, 2),
        ])
def test_normality(mu, n=3):
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    tree = numpy.concatenate(
        orthopy.disk.tree(n, numpy.array([x, y]), mu=mu, symbolic=True)
        )

    # Cartesian integration in sympy is bugged, cf.
    # <https://github.com/sympy/sympy/issues/13816>.
    # Simply transform to polar coordinates for now.
    r = sympy.Symbol('r')
    phi = sympy.Symbol('phi')

    one_half = sympy.Rational(1, 2)
    W = (mu + one_half) * (1-r**2)**(mu-one_half)
    for val in tree:
        val_r_phi = val.subs([(x, r*sympy.cos(phi)), (y, r*sympy.sin(phi))])
        assert sympy.integrate(
            W * r * val_r_phi**2,
            (r, 0, 1), (phi, 0, 2*sympy.pi)
            ) == 1
    return


def test_show(n=4, r=3):
    def f(X):
        return orthopy.disk.tree(n, X)[n][r]

    orthopy.disk.show(f)
    # orthopy.disk.plot(f, lcar=2.0e-2)
    # import matplotlib.pyplot as plt
    # plt.savefig('disk.png', transparent=True)
    return


if __name__ == '__main__':
    test_show()
