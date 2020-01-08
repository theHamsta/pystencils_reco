# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import sympy


def gaussian(input_symbols, sigma=1):
    """gaussian

    :param input_symbols: Input symbols for each dimension
    :param sigma: Standard deviation or covariance matrix
    """

    ndim = len(input_symbols)
    if not isinstance(sigma, sympy.Matrix):
        covariance_matrix = sympy.Matrix(sympy.Identity(ndim)) * sigma * sigma
    else:
        covariance_matrix = sigma

    x = sympy.Matrix(input_symbols)
    return sympy.sqrt(sympy.pi**ndim / covariance_matrix.det()) * \
        sympy.exp((-x.transpose() @ covariance_matrix.inv() @ x)[0, 0])


def squared_difference(x, y):
    """squared_difference

    :param x: sympy.Symbol
    :param y: sympy.Symbol
    """

    return (x-y)*(x-y)


def absolute_difference(x, y):
    """absolute_difference

    :param x: sympy.Symbol
    :param y: sympy.Symbol
    """

    return sympy.Abs(x-y)
