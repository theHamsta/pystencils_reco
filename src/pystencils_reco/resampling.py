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
import sympy.matrices.dense

import pystencils
from pystencils_reco import AssignmentCollection


def generic_spatial_matrix_transform(input_field, output_field, transform_matrix):
    texture = pystencils.astnodes.TextureCachedField(input_field)

    assignments = AssignmentCollection({
        output_field.center(): texture.at(transform_matrix.inv() @ pystencils.x_staggered_vector)
    })
    assignments.transform_matrix = transform_matrix

    return assignments


def scale_transform(input_field, output_field, scaling_factor):
    """scale_transform

    :param input_field:
    :param output_field:
    :param scaling_factor: Number, sympy.Symbol or list describing the scaling in along each dimension
    """

    if isinstance(scale_transform, Iterable):
        transform_matrix = sympy.matrices.diag(scaling_factor)
    else:
        transform_matrix = sympy.matrices.diag([scaling_factor] * input_field.spatial_dimensions)

    return generic_spatial_matrix_transform(input_field, output_field, transform_matrix)


def rotation_transform(input_field, output_field, rotation_angle, rotation_axis=None):
    if input_field.spatial_dimensions == 3:
        assert rotation_axis is not None, "You must specify a rotation_axis for 3d rotations!"
        get_rotation_matrix = getattr(sympy.matrices.dense, 'rot_axis%i' % rotation_axis)
    elif input_field.spatial_dimensions == 2:
        get_rotation_matrix = sympy.matrices.dense.rot_axis1
    else:
        raise NotImplementedError('Rotations only implemented for 2d and 3d')

    transform_matrix = get_rotation_matrix(rotation_angle)

    return generic_spatial_matrix_transform(input_field, output_field, transform_matrix)
