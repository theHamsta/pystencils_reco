# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""
Implements common resampling operations like rotations and scalings
"""

from collections.abc import Iterable

import sympy

import pystencils
from pystencils_reco import AssignmentCollection


def generic_spatial_matrix_transform(input_field, output_field, transform_matrix, inverse_matrix=None):
    texture = pystencils.astnodes.TextureCachedField(input_field)

    if inverse_matrix is None:
        inverse_matrix = transform_matrix.inv()

    assignments = AssignmentCollection({
        output_field.center(): texture.at(input_field.coordinate_transform.inv() @
                                          (inverse_matrix @ output_field.physical_coordinates_staggered) - input_field.coordinate_origin)
    })
    assignments.transform_matrix = transform_matrix

    return assignments


def scale_transform(input_field, output_field, scaling_factor):
    """scale_transform

    :param input_field:
    :param output_field:
    :param scaling_factor: Number, sympy.Symbol or list describing the scaling in along each dimension
    """

    if isinstance(scaling_factor, Iterable):
        transform_matrix = sympy.diag(*scaling_factor)
    else:
        transform_matrix = sympy.diag(*list([scaling_factor] * input_field.spatial_dimensions))

    return generic_spatial_matrix_transform(input_field, output_field, transform_matrix)


def rotation_transform(input_field, output_field, rotation_angle, rotation_axis=None):
    if input_field.spatial_dimensions == 3:
        assert rotation_axis is not None, "You must specify a rotation_axis for 3d rotations!"
        transform_matrix = getattr(sympy, 'rot_axis%i' % (rotation_axis+1))(rotation_angle)
    elif input_field.spatial_dimensions == 2:
        # 2d rotation is 3d rotation around 3rd axis
        transform_matrix = sympy.rot_axis3(rotation_angle)[:2, :2]
    else:
        raise NotImplementedError('Rotations only implemented for 2d and 3d')

    return generic_spatial_matrix_transform(input_field, output_field, transform_matrix, inverse_matrix=transform_matrix.T)

def resample(input_field, output_field):
    """
    Resample input_field with its coordinate system in terms of the coordinate system of output_field

    :param input_field:
    :param output_field:
    """

    return generic_spatial_matrix_transform(input_field, output_field, sympy.Matrix(sympy.Identity(input_field.spatial_dimensions)))

