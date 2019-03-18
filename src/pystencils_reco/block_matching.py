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


def block_matching_integer_offsets(input_field: Field,
                                   comparision_field: Field,
                                   output_block_scores: Field,
                                   block_stencil,
                                   matching_stencil,
                                   matching_function=pystencils_reco.functions.squared_difference):

    assert output_block_scores.index_dimensions == len(matching_stencil), \
        "Channels/index_dimensions of output_block_scores must match length of matching_stencil"

    assignments = []

    for i, m in enumerate(matching_stencil):
        rhs = 0

        for s in block_stencil:
            rhs += matching_function(input_field(s), comparision_field(s))

        lhs = output_block_scores.center()(i)
        assignment = pystencils.Assignment(lhs, rhs)
        assignments.append(assignment)

    return AssignmentCollection(assignments)
