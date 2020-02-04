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
from typing import Union

import sympy

import pystencils
import pystencils.autodiff
from pystencils.autodiff import AdjointField
from pystencils.data_types import cast_func, create_type
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
        output_field.center:
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

    # assignments._create_autodiff = types.MethodType(create_autodiff, assignments)
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
    if isinstance(rotation_angle, pystencils.Field):
        rotation_angle = rotation_angle.center

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
              interpolation_mode='linear',
              allow_spatial_derivatives=True):

    # def create_autodiff(self, constant_fields=None, **kwargs):
        # backward_assignments = translate(AdjointField(output_field), AdjointField(input_field), -translation)
        # self._autodiff = pystencils.autodiff.AutoDiffOp(
            # assignments, "", backward_assignments=backward_assignments, **kwargs)

    if isinstance(translation, pystencils.Field):
        translation = translation.center_vector

    assignments = AssignmentCollection(
        {
            output_field.center: input_field.interpolated_access(
                input_field.physical_to_index(output_field.physical_coordinates - translation), interpolation_mode)
        })

    # if not allow_spatial_derivatives:
    # assignments._create_autodiff = types.MethodType(create_autodiff, assignments)
    return assignments


@crazy
def upsample(input: {'field_type': pystencils.field.FieldType.CUSTOM},
             result,
             factor):

    ndim = input.spatial_dimensions
    here = pystencils.x_vector(ndim)

    assignments = AssignmentCollection(
        {result.center:
            pystencils.astnodes.ConditionalFieldAccess(
                input.absolute_access(tuple(cast_func(sympy.S(1) / factor * h,
                                                      create_type('int64')) for h in here), ()),
                sympy.Or(*[s % cast_func(factor, 'int64') > 0 for s in here]))
         })

    def create_autodiff(self, constant_fields=None, **kwargs):
        backward_assignments = downsample(AdjointField(result), AdjointField(input), factor)
        self._autodiff = pystencils.autodiff.AutoDiffOp(
            assignments, "", backward_assignments=backward_assignments, **kwargs)

    assignments._create_autodiff = types.MethodType(create_autodiff, assignments)
    return assignments


@crazy
def downsample(input: {'field_type': pystencils.field.FieldType.CUSTOM},
               result,
               factor):
    assert input.spatial_dimensions == result.spatial_dimensions
    assert input.index_shape == result.index_shape

    ndim = input.spatial_dimensions

    assignments = AssignmentCollection({result.center:
                                        input.absolute_access(factor * pystencils.x_vector(ndim), ())})

    def create_autodiff(self, constant_fields=None, **kwargs):
        backward_assignments = upsample(AdjointField(result), AdjointField(input), factor)
        self._autodiff = pystencils.autodiff.AutoDiffOp(
            assignments, "", backward_assignments=backward_assignments, **kwargs)

    assignments._create_autodiff = types.MethodType(create_autodiff, assignments)

    return assignments


@crazy
def resample_to_shape(input,
                      spatial_shape: Union[tuple, pystencils.Field],
                      ):
    if hasattr(spatial_shape, 'spatial_shape'):
        spatial_shape = spatial_shape.spatial_shape

    output_field = pystencils.Field.create_fixed_size(
        'output', spatial_shape + input.index_shape, input.index_dimensions, input.dtype.numpy_dtype)
    output_field.coordinate_transform = sympy.DiagMatrix(sympy.Matrix([input.spatial_shape[i] / s
                                                                       for i, s in enumerate(spatial_shape)]))
    return resample(input, output_field)
