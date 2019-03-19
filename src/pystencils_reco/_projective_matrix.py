# -*- coding: utf-8 -*-
#
# Copyright © 2019 stephan <stephan@stephan-Z87-DS3H>
#
# Distributed under terms of the GPLv3 license.

"""
Implements class ProjectiveMatrix
"""
import sympy


class ProjectiveMatrix(object):
    """docstring for ProjectiveMatrix"""

    def __init__(self, matrix):
        if isinstance(matrix, ProjectiveMatrix):
            matrix = matrix.matrix
        self.matrix = sympy.Matrix(matrix)

    def __matmul__(self, vector):
        vector = sympy.Matrix(vector)
        if vector.rows == self.matrix.cols:
            return self.matrix @ vector

        if vector.rows + 1 == self.matrix.cols:
            lifted_vector = sympy.Matrix([vector, [1]])
            result = self.matrix @ lifted_vector
            normalized_result = sympy.Matrix(result[:-1]) / result[-1]
            return normalized_result

        if vector.rows == self.matrix.cols + 1:
            lifted_matrix = self.matrix.row_insert(-1, sympy.Matrix([0]*self.matrix.cols).T)
            lifted_matrix = lifted_matrix.col_insert(-1, sympy.Matrix([0]*lifted_matrix.rows))
            lifted_matrix[-1, -1] = 1
            result = lifted_matrix @ vector
            return result

        raise NotImplementedError(
            "Can only multiply vectors with same number of rows or one less with ProjectiveMatrix")

    def nullspace(self):
        return self.matrix.nullspace()
