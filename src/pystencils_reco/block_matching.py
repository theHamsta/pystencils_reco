# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import uuid

import sympy
from tqdm import tqdm

import pystencils
import pystencils_reco.functions
from pystencils import Field
from pystencils.astnodes import ForEach, Select, address_of
from pystencils_reco import AssignmentCollection, crazy


@crazy
def block_matching_integer_offsets_unrolled(input_field: Field,
                                            comparision_field: Field,
                                            output_block_scores: Field,
                                            block_stencil,
                                            matching_stencil,
                                            matching_function=pystencils_reco.functions.squared_difference):

    assert output_block_scores.index_dimensions == 1, \
        "output_block_scores must have channels equal to the length of matching_stencil"
    assert output_block_scores.index_shape[0] == len(matching_stencil), \
        "Channels/index_dimensions of output_block_scores must match length of matching_stencil"

    assignments = []

    for i, m in tqdm(enumerate(matching_stencil), total=len(matching_stencil)):
        rhs = []

        for s in block_stencil:
            shifted = tuple(i + j for i, j in zip(s, m))
            rhs.append(matching_function(input_field[s], comparision_field[shifted]))

        lhs = output_block_scores(i)
        assignment = pystencils.Assignment(lhs, sympy.Add(*rhs))
        assignments.append(assignment)

    return AssignmentCollection(assignments, perform_cse=False)


@crazy
def single_block_matching(input_field: Field,
                          comparision_field: Field,
                          output_block_scores: Field,
                          block_stencil,
                          matching_offset,
                          match_index,
                          matching_function=pystencils_reco.functions.squared_difference,
                          ):

    assignments = []

    i, m = match_index, matching_offset
    rhs = []

    for s in block_stencil:
        shifted = tuple(i + j for i, j in zip(s, m))
        rhs.append(matching_function(input_field[s], comparision_field[shifted]))

    lhs = output_block_scores(i)
    assignment = pystencils.Assignment(lhs, sympy.Add(*rhs))
    assignments.append(assignment)

    return AssignmentCollection(assignments, perform_cse=False)


@crazy
def block_matching_integer_offsets(input_field: Field,
                                   comparision_field: Field,
                                   output_block_scores: Field,
                                   block_stencil,
                                   matching_stencil,
                                   compilation_target,
                                   matching_function=pystencils_reco.functions.squared_difference,
                                   **compilation_kwargs):
    max_offset = max(max(o) for o in matching_stencil)
    max_offset += max(max(o) for o in block_stencil)

    offset = pystencils.typed_symbols('_o:%i' % input_field.spatial_dimensions, 'int32')
    i = pystencils.typed_symbols('_i', 'int32')
    block_matching = single_block_matching(input_field, comparision_field,
                                           output_block_scores, block_stencil, offset, i, matching_function)
    ast = pystencils.create_kernel(block_matching, target=compilation_target,
                                   data_type=input_field.dtype,
                                   ghost_layers=max_offset,
                                   **compilation_kwargs)
    # TODO: determine necessary ghost_layers
    # TODO: move into LoopOverCoordinate body (better performance on CPU?)
    ast._body = ForEach(ast.body, offset, matching_stencil, i)
    return pystencils.make_python_function(ast, target=compilation_target)


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

    offset = pystencils.typed_symbols('_o:%i' % patch_input_field.spatial_dimensions, 'int32')
    copies = []

    assert destination_field.index_dimensions == 2
    assert destination_field.index_shape[-1] == len(block_stencil)

    n, nth_hit = pystencils.typed_symbols('_n, nth_hit', 'int32')
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
    return pystencils.make_python_function(ast, target=compilation_target)


@crazy
def aggregate(block_scores: Field,
              patch_input_field: Field,
              destination_field: Field,
              block_stencil,
              matching_stencil,
              threshold,
              max_selected,
              compilation_target,
              **compilation_kwargs):

    max_offset = max(max(o) for o in matching_stencil)
    max_offset += max(max(o) for o in block_stencil)

    offset = pystencils.typed_symbols('_o:%i' % patch_input_field.spatial_dimensions, 'int32')
    copies = []

    assert destination_field.index_dimensions == 2
    assert destination_field.index_shape[-1] == len(block_stencil)

    n, nth_hit = pystencils.typed_symbols('_n, nth_hit', 'int32')
    for i, s in enumerate(block_stencil):
        shifted = tuple(s + o for s, o in zip(offset, s))
        copies.append(pystencils.Assignment(pystencils.typed_symbols('dummy%s' % uuid.uuid4().hex, 'bool'),
                                            sympy.Function('atomicAdd')(address_of(patch_input_field[shifted]),
                                                                        destination_field.center(nth_hit, i))))

    assignments = AssignmentCollection(copies)
    ast = pystencils.create_kernel(assignments, target=compilation_target,
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
    return pystencils.make_python_function(ast, target=compilation_target)



    # assert block_scores.index_dimensions == 1, \
    # "output_block_scores must have channels equal to the length of matching_stencil"
    # assert output_block_scores.index_shape[0] == len(matching_stencil), \
    # "Channels/index_dimensions of output_block_scores must match length of matching_stencil"

    # assignments = []

    # for i, m in enumerate(matching_stencil):
    # rhs = 0

    # for s in block_stencil:
    # shifted = tuple(i + j for i, j in zip(s, m))
    # rhs += matching_function(input_field[s], comparision_field[shifted])

    # lhs = output_block_scores(i)
    # assignment = pystencils.Assignment(lhs, rhs)
    # assignments.append(assignment)

    # return AssignmentCollection(assignments, perform_cse=False)
