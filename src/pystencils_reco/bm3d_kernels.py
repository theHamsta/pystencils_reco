# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import uuid

import sympy

import pystencils
import pystencils_reco
from pystencils import Field
from pystencils.data_types import address_of, create_type
from pystencils_reco import AssignmentCollection, crazy
from pystencils_reco.astnodes import Select


def _get_dummy_symbol(dtype=None):
    return pystencils_reco.typed_symbols('dummy%s' % uuid.uuid4().hex, dtype or create_type('bool'))


@crazy
def collect_patches(block_scores: Field,
                    patch_input_field: Field,
                    destination_field: Field,
                    block_stencil,
                    matching_stencil,
                    threshold,
                    max_selected,
                    compilation_target,
                    **compilation_kwargs
                    ):
    max_offset = max(max(o) for o in matching_stencil)
    max_offset += max(max(o) for o in block_stencil)

    offset = pystencils_reco.typed_symbols('_o:%i' % patch_input_field.spatial_dimensions, 'int32')
    copies = []

    assert destination_field.index_dimensions == 2
    assert destination_field.index_shape[-1] == len(block_stencil)

    n, nth_hit = pystencils_reco.typed_symbols('_n, nth_hit', 'int32')
    for i, s in enumerate(block_stencil):
        shifted = tuple(s + o for s, o in zip(offset, s))
        copies.append(pystencils.Assignment(destination_field.center(nth_hit, i), patch_input_field[shifted]))

    assignments = AssignmentCollection(copies)
    ast = pystencils.create_kernel(assignments, target=compilation_target,
                                   data_type=patch_input_field.dtype,
                                   ghost_layers=max_offset,
                                   **compilation_kwargs)
    # TODO move select on per coordinate level
    ast._body = Select(ast.body,
                       what=offset,
                       from_iterable=matching_stencil,
                       predicate=block_scores.center(n) < threshold,
                       counter_symbol=n,
                       hit_counter_symbol=nth_hit,
                       max_selected=max_selected,
                       compilation_target=compilation_target)
    return ast.compile()


@crazy
def aggregate(block_scores: Field,
              patch_input_field: Field,
              destination_field: Field,
              block_stencil,
              matching_stencil,
              threshold,
              max_selected,
              compilation_target,
              patch_weights: Field = None,
              accumulated_weights: Field = None,
              **compilation_kwargs):

    max_offset = max(max(o) for o in matching_stencil)
    max_offset += max(max(o) for o in block_stencil)

    offset = pystencils_reco.typed_symbols('_o:%i' % patch_input_field.spatial_dimensions, 'int32')
    copies = []

    assert destination_field.index_dimensions == 2
    assert destination_field.index_shape[-1] == len(block_stencil)

    n, nth_hit = pystencils_reco.typed_symbols('_n, nth_hit', 'int32')
    for i, s in enumerate(block_stencil):
        shifted = tuple(s + o for s, o in zip(offset, s))
        weight = patch_weights.center(nth_hit) if patch_weights else 1

        assignment = pystencils.Assignment(_get_dummy_symbol(),
                                           sympy.Function('atomicAdd')(address_of(patch_input_field[shifted]),
                                                                       weight * destination_field.center(nth_hit, i)))
        copies.append(assignment)
        if accumulated_weights:
            assignment = pystencils.Assignment(_get_dummy_symbol(),
                                               sympy.Function('atomicAdd')(
                                                   address_of(accumulated_weights[shifted]), weight))
            copies.append(assignment)

    assignments = AssignmentCollection(copies)
    ast = pystencils.create_kernel(assignments,
                                   target=compilation_target,
                                   data_type=patch_input_field.dtype,
                                   ghost_layers=max_offset,
                                   **compilation_kwargs)

    ast._body = Select(ast.body,
                       what=offset,
                       from_iterable=matching_stencil,
                       predicate=block_scores.center(n) < threshold,
                       counter_symbol=n,
                       hit_counter_symbol=nth_hit,
                       compilation_target=compilation_target,
                       max_selected=max_selected)
    return ast.compile()


@crazy
def hard_thresholding(complex_field: Field, output_weight_field, threshold):
    assert complex_field.index_dimensions == 3
    assert output_weight_field.index_dimensions == 1

    assignments = []

    for stack_index in range(complex_field.index_shape[0]):
        num_nonzeros = []
        for patch_index in range(complex_field.index_shape[1]):

            magnitude = sum(complex_field.center(stack_index, patch_index, i) ** 2 for i in (0, 1))
            assignments.extend(
                pystencils.Assignment(complex_field.center(stack_index, patch_index, i),
                                      sympy.Piecewise(
                    (complex_field.center(stack_index, patch_index, i),
                     magnitude > threshold ** 2), (0, True)))
                for i in (0, 1)
            )
            num_nonzeros.append(sympy.Piecewise((1, magnitude > threshold ** 2), (0, True)))

        assignments.append(pystencils.Assignment(
            output_weight_field.center(stack_index), sympy.Add(*num_nonzeros)
        ))

    return AssignmentCollection(assignments)


@crazy
def wiener_filtering(complex_field: Field, output_weight_field: Field, sigma):
    assert complex_field.index_dimensions == 3
    assert output_weight_field.index_dimensions == 1

    assignments = []

    norm_factor = complex_field.index_shape[0] * complex_field.index_shape[1]
    wiener_sum = []

    for stack_index in range(complex_field.index_shape[0]):
        for patch_index in range(complex_field.index_shape[1]):

            magnitude = sum(complex_field.center(stack_index, patch_index, i) ** 2 for i in (0, 1))
            val = magnitude / norm_factor  # implementation differ whether to apply norm_factor on val on wien
            wien = val / (val + sigma * sigma)

            wiener_sum.append(wien**2)

            assignments.extend(
                pystencils.Assignment(complex_field.center(stack_index, patch_index, i),
                                      complex_field.center(stack_index, patch_index, i) * wien)
                for i in (0, 1)
            )

        assignments.append(pystencils.Assignment(
            output_weight_field.center(stack_index), 1 / sympy.Add(*wiener_sum)
        ))

    return AssignmentCollection(assignments)


@crazy
def apply_wieners(complex_field: Field, wieners: Field, output_weight_field: Field):
    assert complex_field.index_dimensions == 3
    assert wieners.index_dimensions == 2
    assert output_weight_field.index_dimensions == 1

    assignments = []
    wiener_sum = []

    for stack_index in range(complex_field.index_shape[0]):
        for patch_index in range(complex_field.index_shape[1]):

            wien = wieners(stack_index, patch_index)

            wiener_sum.append(wien**2)

            assignments.extend(
                pystencils.Assignment(complex_field.center(stack_index, patch_index, i),
                                      complex_field.center(stack_index, patch_index, i) * wien)
                for i in (0, 1)
            )

        assignments.append(pystencils.Assignment(
            output_weight_field.center(stack_index), 1 / sympy.Add(*wiener_sum)
        ))

    return AssignmentCollection(assignments)


@crazy
def calc_wiener_coefficients(complex_field: Field, output_wieners: Field, sigma):
    assert complex_field.index_dimensions == 3
    assert output_wieners.index_dimensions == 2

    assignments = []
    norm_factor = complex_field.index_shape[0] * complex_field.index_shape[1]

    for stack_index in range(complex_field.index_shape[0]):
        for patch_index in range(complex_field.index_shape[1]):

            magnitude = sum(complex_field.center(stack_index, patch_index, i) ** 2 for i in (0, 1))
            val = magnitude / norm_factor
            wien = val / (val + sigma * sigma)

            assignments.append(
                pystencils.Assignment(output_wieners.center(stack_index, patch_index), wien)
            )

    return AssignmentCollection(assignments)
