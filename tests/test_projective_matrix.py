# -*- coding: utf-8 -*-
#
# Copyright © 2019 stephan <stephan@stephan-Z87-DS3H>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from pystencils_reco import ProjectiveMatrix
import sympy


def test_projective_matrix():
    matrix = ProjectiveMatrix([[2, 3, 5], [3, 2, 4], [3, 2, 4]])
    vector = sympy.Matrix([2, 3])
    result = matrix @ vector
    print(result)

    matrix = ProjectiveMatrix([[2, 3], [3, 2]])
    vector = sympy.Matrix([2, 3])
    result = matrix @ vector
    print(result)


def main():
    test_projective_matrix()


if __name__ == '__main__':
    main()
