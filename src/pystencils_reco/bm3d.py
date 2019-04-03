# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import sympy

import pystencils
import pystencils_reco
from pystencils import Field
from pystencils_reco import crazy
from pystencils_reco.block_matching import (aggregate,
                                            block_matching_integer_offsets,
                                            collect_patches)


@crazy
def _hard_thresholding(complex_field: Field, output_weight_field, threshold):
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
                     magnitude > threshold ** 2),  (0, True)))
                for i in (0, 1)
            )
            num_nonzeros.append(sympy.Piecewise((1, magnitude > threshold ** 2), (0, True)))

        assignments.append(pystencils.Assignment(
            output_weight_field.center(stack_index), 1 / sympy.Add(*num_nonzeros)
        ))

    return pystencils_reco.AssignmentCollection(assignments)


@crazy
def _wiener_filtering(complex_field: Field, output_weight_field: Field, sigma):
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

    return pystencils_reco.AssignmentCollection(assignments)


@crazy
def _apply_wieners(complex_field: Field, wieners: Field, output_weight_field: Field):
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

    return pystencils_reco.AssignmentCollection(assignments)


@crazy
def _get_wieners(complex_field: Field, output_wieners: Field, sigma):
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

    return pystencils_reco.AssignmentCollection(assignments)


class Bm3d:
    """docstring for Bm3d"""

    def __init__(self, input: Field,
                 output: Field,
                 block_stencil,
                 matching_stencil,
                 compilation_target,
                 max_block_matches,
                 blockmatching_threshold,
                 hard_threshold,
                 matching_function=pystencils_reco.functions.squared_difference,
                 wiener_sigma=None,
                 **compilation_kwargs):
        matching_stencil = sorted(matching_stencil, key=lambda o: sum(abs(o) for o in o))

        input_field = pystencils_reco._crazy_decorator.coerce_to_field('input_field', input)
        output_field = pystencils_reco._crazy_decorator.coerce_to_field('output_field', output)

        block_scores_shape = output_field.shape + (len(matching_stencil),)
        block_scores = Field.create_fixed_size('block_scores',
                                               block_scores_shape,
                                               index_dimensions=1,
                                               dtype=input_field.dtype.numpy_dtype)
        self.block_scores = block_scores

        block_matched_shape = input_field.shape + (max_block_matches, len(block_stencil))
        block_matched_field = Field.create_fixed_size('block_matched',
                                                      block_matched_shape,
                                                      index_dimensions=2,
                                                      dtype=input_field.dtype.numpy_dtype)
        self.block_matched_field = block_matched_field

        self.block_matching = block_matching_integer_offsets(input,
                                                             input,
                                                             block_scores,
                                                             block_stencil,
                                                             matching_stencil,
                                                             compilation_target,
                                                             matching_function,
                                                             **compilation_kwargs)
        self.collect_patches = collect_patches(block_scores,
                                               input,
                                               block_matched_field,
                                               block_stencil,
                                               matching_stencil,
                                               blockmatching_threshold,
                                               max_block_matches,
                                               compilation_target,
                                               **compilation_kwargs)
        complex_field = Field.create_fixed_size('complex_field',
                                                block_matched_shape + (2,),
                                                index_dimensions=3,
                                                dtype=input_field.dtype.numpy_dtype)
        group_weights = Field.create_fixed_size('group_weights',
                                                block_scores_shape,
                                                index_dimensions=1,
                                                dtype=input_field.dtype.numpy_dtype)
        self.complex_field = complex_field
        self.group_weights = group_weights
        self.hard_thresholding = _hard_thresholding(
            complex_field, group_weights, hard_threshold).compile(compilation_target)
        if not wiener_sigma:
            wiener_sigma = pystencils.typed_symbols('wiener_sigma', input_field.dtype.numpy_dtype)
        self.wiener_filtering = _wiener_filtering(
            complex_field, group_weights, wiener_sigma).compile(compilation_target)
        wiener_coefficients = Field.create_fixed_size('wiener_coefficients',
                                                      block_matched_shape,
                                                      index_dimensions=2,
                                                      dtype=input_field.dtype.numpy_dtype)
        self.get_wieners = _get_wieners(complex_field, wiener_coefficients,
                                        wiener_sigma).compile(compilation_target)
        self.apply_wieners = _apply_wieners(complex_field, wiener_coefficients,
                                            group_weights).compile(compilation_target)

        self.aggregate = aggregate(block_scores,
                                   output,
                                   block_matched_field,
                                   block_stencil,
                                   matching_stencil,
                                   blockmatching_threshold,
                                   max_block_matches,
                                   compilation_target,
                                   **compilation_kwargs)
