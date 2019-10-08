# -*- coding: utf-8 -*-
#
# Copyright © 2019 stephan <stephan@stephan-Z87-DS3H>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import sympy
from pystencils_reco import ProjectiveMatrix


def test_projective_matrix():
    matrix = ProjectiveMatrix([[2, 3, 5], [3, 2, 4]])
    vector = sympy.Matrix([2, 3])
    result = matrix @ vector
    print(result)

    matrix = ProjectiveMatrix([[2, 3], [3, 2]])
    vector = sympy.Matrix([2, 3])
    result = matrix @ vector
    print(result)

    print('Perspective projection')
    projection_matrix = ProjectiveMatrix([[-289.0098977737411, -1205.2274801832275, 0.0, 186000.0],
                                          [-239.9634468375339, - 4.188577544948043, 1200.0, 144000.0],
                                          [-0.9998476951563913, -0.01745240643728351, 0.0, 600.0]])
    x = sympy.Matrix(sympy.symbols('x:3'))
    u = sympy.Matrix(sympy.symbols('u:2'))
    eqn = projection_matrix @ x - u
    ray_equations = sympy.solve(eqn, x)
    for k, v in ray_equations.items():
        print(f"{k}: {v}")

    print('Orthogonal projection')
    projection_matrix = ProjectiveMatrix([[1, 0, 0],
                                          [0, 1, 0]])
    x = sympy.Matrix(sympy.symbols('x:3'))
    u = sympy.Matrix(sympy.symbols('u:2'))
    eqn = projection_matrix @ x - u
    ray_equations = sympy.solve(eqn, x)
    for k, v in ray_equations.items():
        print(f"{k}: {v}")

    print(projection_matrix.nullspace())
