# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""
Implements common resampling operations like rotations and scalings
"""

import types
from collections.abc import Iterable

import sympy

import pystencils
import pystencils.autodiff
from pystencils.autodiff import AdjointField
from pystencils_reco import AssignmentCollection, crazy


@crazy
def generic_spatial_matrix_transform(input_field,
                                     output_field,
                                     transform_matrix,
                                     inverse_matrix=None,
                                     interpolation_mode='linear'):
    texture = pystencils.interpolation_astnodes.Interpolator(input_field,
                                                             interpolation_mode=interpolation_mode)

    if inverse_matrix is None:
        inverse_matrix = transform_matrix.inv()

    # output_coordinate = input_field.coordinate_transform.inv() @ (
        # inverse_matrix @ output_field.physical_coordinates_staggered) - input_field.coordinate_origin
    output_coordinate = input_field.physical_to_index(
        inverse_matrix @ output_field.physical_coordinates, staggered=False)

    assignments = AssignmentCollection({
        output_field.center():
        texture.at(output_coordinate)
    })

    def create_autodiff(self, constant_fields=None, **kwargs):
        assignments.transform_matrix = transform_matrix

        texture = pystencils.interpolation_astnodes.Interpolator(
            AdjointField(output_field), interpolation_mode=interpolation_mode)
        output_coordinate = output_field.physical_to_index(
            transform_matrix @ input_field.physical_coordinates, staggered=False)
        backward_assignments = AssignmentCollection({
            AdjointField(input_field).center(): texture.at(output_coordinate)
        })
        self._autodiff = pystencils.autodiff.AutoDiffOp(
            assignments, "", backward_assignments=backward_assignments, **kwargs)

    assignments._create_autodiff = types.MethodType(create_autodiff, assignments)
    return assignments


@crazy
def scale_transform(input_field, output_field, scaling_factor, interpolation_mode='linear'):
    """scale_transform

    :param input_field:
    :param output_field:
    :param scaling_factor: Number, sympy.Symbol or list describing the scaling in along each dimension
    """

    if isinstance(scaling_factor, Iterable):
        transform_matrix = sympy.diag(*scaling_factor)
    else:
        transform_matrix = sympy.diag(*list([scaling_factor] * input_field.spatial_dimensions))

    return generic_spatial_matrix_transform(input_field,
                                            output_field,
                                            transform_matrix,
                                            interpolation_mode=interpolation_mode)


@crazy
def rotation_transform(input_field,
                       output_field,
                       rotation_angle,
                       rotation_axis=None,
                       interpolation_mode='linear'):
    if input_field.spatial_dimensions == 3:
        assert rotation_axis is not None, "You must specify a rotation_axis for 3d rotations!"
        transform_matrix = getattr(sympy, 'rot_axis%i' % (rotation_axis + 1))(rotation_angle)
    elif input_field.spatial_dimensions == 2:
        # 2d rotation is 3d rotation around 3rd axis
        transform_matrix = sympy.rot_axis3(rotation_angle)[:2, :2]
    else:
        raise NotImplementedError('Rotations only implemented for 2d and 3d')

    return generic_spatial_matrix_transform(input_field,
                                            output_field,
                                            transform_matrix,
                                            inverse_matrix=transform_matrix.T,
                                            interpolation_mode=interpolation_mode)


@crazy
def resample(input_field, output_field, interpolation_mode='linear'):
    """
    Resample input_field with its coordinate system in terms of the coordinate system of output_field

    :param input_field:
    :param output_field:
    """

    return generic_spatial_matrix_transform(input_field,
                                            output_field,
                                            sympy.Matrix(sympy.Identity(input_field.spatial_dimensions)),
                                            interpolation_mode=interpolation_mode)


@crazy
def translate(input_field: pystencils.Field,
              output_field: pystencils.Field,
              translation,
              interpolation_mode='linear'):

    return {
        output_field.center: input_field.interpolated_access(
            input_field.physical_to_index(output_field.physical_coordinates - translation), interpolation_mode)
    }
