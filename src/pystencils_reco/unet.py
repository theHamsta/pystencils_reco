# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import itertools

import sympy

import pystencils
from pystencils.data_types import cast_func, create_type
from pystencils.field import FieldType
from pystencils_reco import crazy


@crazy
def relu(input, result):
    assert input.spatial_dimensions == result.spatial_dimensions
    assert input.index_shape == result.index_shape
    assignments = []
    for i in range(input.index_shape[0]):
        assignments.append(pystencils.Assignment(result.center(i), sympy.Max(input.center(i), 0)))
    return assignments


@crazy
def max_pooling(input: {'field_type': FieldType.CUSTOM},
                result):
    assert input.spatial_dimensions == result.spatial_dimensions
    assert input.index_shape == result.index_shape
    assignments = []

    ndim = input.spatial_dimensions

    offsets = itertools.product((0, 1), repeat=ndim)
    assignments.append(
        pystencils.Assignment(result.center,
                              sympy.Max(*[input.absolute_access(2 * pystencils.x_vector(ndim) + sympy.Matrix(offset), ())  # noqa
                                          for offset in offsets])
                              )
    )

    return assignments


@crazy
def max_pooling_channels(input: {'index_dimensions': 1, 'field_type': FieldType.CUSTOM},
                         result: {'index_dimensions': 1}):
    assert input.spatial_dimensions == result.spatial_dimensions
    assert input.index_shape == result.index_shape
    assignments = []

    ndim = input.spatial_dimensions

    for i in range(result.index_shape[0]):
        offsets = itertools.product((0, 1), repeat=ndim)
        assignments.append(
            pystencils.Assignment(result.center(i),
                                  sympy.Max(*[input.absolute_access(2 * pystencils.x_vector(ndim) + sympy.Matrix(offset), (i,))  # noqa
                                              for offset in offsets])
                                  )
        )

    return assignments


@crazy
def channel_convolution(input: {'index_dimensions': 1, 'field_type': FieldType.CUSTOM},
                        stencil: {'index_dimensions': 2, 'field_type': FieldType.CUSTOM},
                        result: {'index_dimensions': 1}):
    assert input.index_shape[0] == stencil.index_shape[0]
    assert result.index_shape[0] == stencil.index_shape[1]
    assert input.spatial_dimensions == result.spatial_dimensions

    assignments = []

    for i in range(result.index_shape[0]):
        for j in range(input.index_shape[0]):
            rhs = []
            for offset in itertools.product(*[range(-s//2+1, s//2 + 1) for s in stencil.spatial_shape]):
                rhs.append(input.__getitem__(offset)(j) *
                           stencil.absolute_access(tuple(o + s//2 for o, s in zip(offset, stencil.spatial_shape)), (j, i)))  # noqa
            assignment = pystencils.Assignment(result.center(i), sympy.Add(*rhs))
            assignments.append(assignment)
    return assignments


@crazy
def convolution(input: {'field_type': FieldType.CUSTOM},
                stencil: {'field_type': FieldType.CUSTOM},
                result):
    assignments = []

    rhs = []
    for offset in itertools.product(*[range(-s//2+1, s//2 + 1) for s in stencil.spatial_shape]):
        rhs.append(input.__getitem__(offset) * stencil.absolute_access(tuple(o + s//2 for o, s in zip(offset, stencil.spatial_shape)), ()))  # noqa
    assignment = pystencils.Assignment(result.center, sympy.Add(*rhs))
    assignments.append(assignment)
    return assignments


@crazy
def upsample(input: {'field_type': FieldType.CUSTOM},
             result,
             sampling_factor=2):

    assert input.spatial_dimensions == result.spatial_dimensions
    assert input.field_type == FieldType.CUSTOM \
        or result.spatial_shape == tuple([2 * x for x in input.spatial_shape])
    assert input.index_shape == result.index_shape
    assignments = []
    ndim = input.spatial_dimensions

    for i in range(result.index_shape[0]):
        assignments.append(
            pystencils.Assignment(result.center(i),
                                  input.absolute_access(
                sympy.Matrix(tuple([cast_func(x // sampling_factor, create_type("int")) for x in pystencils.x_vector(ndim)])), (i,)))
        )
    return assignments
