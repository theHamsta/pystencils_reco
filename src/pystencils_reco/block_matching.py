# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pystencils
import pystencils_reco.functions
from pystencils import Field
from pystencils_reco import AssignmentCollection
import sympy
from tqdm import tqdm

def block_matching_integer_offsets(input_field: Field,
                                   comparision_field: Field,
                                   output_block_scores: Field,
                                   block_stencil,
                                   matching_stencil,
                                   matching_function=pystencils_reco.functions.squared_difference,
                                   collect_field: Field = None,
                                   collect_criterion=None):

    assert output_block_scores.index_dimensions == 1, \
        "output_block_scores must have channels equal to the length of matching_stencil"
    assert output_block_scores.index_shape[0] == len(matching_stencil), \
        "Channels/index_dimensions of output_block_scores must match length of matching_stencil"

    assignments = []

    for i, m in tqdm(enumerate(matching_stencil), total=len(matching_stencil)):
        rhs = []

        for s in block_stencil:
            shifted = tuple(i+j for i, j in zip(s, m))
            rhs.append(matching_function(input_field[s], comparision_field[shifted]))

        lhs = output_block_scores(i)
        assignment = pystencils.Assignment(lhs, sympy.Add(*rhs))
        assignments.append(assignment)

    return AssignmentCollection(assignments, perform_cse=False)


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


def collect_patches(block_scores: Field,
                    patches: Field,
                    block_stencil,
                    matching_stencil,
                    matching_function=pystencils_reco.functions.squared_difference):

    assert block_scores.index_dimensions == 1, \
        "output_block_scores must have channels equal to the length of matching_stencil"
    assert output_block_scores.index_shape[0] == len(matching_stencil), \
        "Channels/index_dimensions of output_block_scores must match length of matching_stencil"

    assignments = []

    for i, m in enumerate(matching_stencil):
        rhs = 0

        for s in block_stencil:
            shifted = tuple(i + j for i, j in zip(s, m))
            rhs += matching_function(input_field[s], comparision_field[shifted])

        lhs = output_block_scores(i)
        assignment = pystencils.Assignment(lhs, rhs)
        assignments.append(assignment)

    return AssignmentCollection(assignments, perform_cse=False)

