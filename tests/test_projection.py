# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import sympy

import pystencils
from pystencils_reco.projection import forward_projection


def test_projection():
    volume = pystencils.fields('volume: float32[100,200,300]')
    projections = pystencils.fields('projections: float32[2D]')

    projection_matrix = sympy.Matrix([[-289.0098977737411, -1205.2274801832275, 0.0, 186000.0],
                                      [-239.9634468375339, - 4.188577544948043, 1200.0, 144000.0],
                                      [-0.9998476951563913, -0.01745240643728351, 0.0, 600.0]])

    kernel = forward_projection(volume, projections, projection_matrix).compile(target='gpu')
    print(kernel.code)

    # a = sympy.Symbol('a')
    # projection_matrix = sympy.Matrix([[-289.0098977737411, -1205.2274801832275, 0.0, 186000.0],
    # [a, - 4.188577544948043, 1200.0, 144000.0],
    # [-0.9998476951563913, -0.01745240643728351, 0.0, 600.0]])

    # kernel = forward_projection(volume, projections, projection_matrix).compile(target='gpu')
    # print(kernel.code)

    a = sympy.symbols('a:12')
    a = [pystencils.TypedSymbol(s.name, 'float32') for s in a]
    A = sympy.Matrix(3, 4, lambda i, j: a[i*4+j])
    # A = sympy.MatrixSymbol('A', 3, 4)
    projection_matrix = A
    kernel = forward_projection(volume, projections, projection_matrix).compile(target='gpu')
    print(kernel.code)


def main():
    test_projection()


if __name__ == '__main__':
    main()
