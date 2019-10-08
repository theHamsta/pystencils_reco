# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pystencils_reco.functions
from sympy import Matrix, Symbol, symbols


def test_functions():
    x, y, z = symbols('x,y,z')
    gaussian = pystencils_reco.functions.gaussian((x, y, z), sigma=1)

    print(gaussian)

    sigmas = [0.1, 0.2, 0.3]
    gaussian = pystencils_reco.functions.gaussian(
        (x, y, z), sigma=Matrix(3, 3, lambda i, j: sigmas[i] if i == j else 0))
    print(gaussian)

    sigmas = [0.1, Symbol('foo'), 0.3]
    gaussian = pystencils_reco.functions.gaussian(
        (x, y, z), sigma=Matrix(3, 3, lambda i, j: sigmas[i] if i == j else 0))
    print(gaussian)
