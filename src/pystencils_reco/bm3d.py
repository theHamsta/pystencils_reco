# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pystencils_reco
from pystencils import Field
from pystencils_reco import crazy
from pystencils_reco.block_matching import (aggregate,
                                            block_matching_integer_offsets,
                                            collect_patches)


class Bm3d:
    """docstring for Bm3d"""

    def __init__(self, input: Field,
                 output: Field,
                 block_stencil,
                 matching_stencil,
                 compilation_target,
                 max_block_matches,
                 threshold,
                 matching_function=pystencils_reco.functions.squared_difference,
                 **compilation_kwargs):

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
                                               threshold,
                                               max_block_matches,
                                               compilation_target,
                                               **compilation_kwargs)
        self.aggregate = aggregate(block_scores,
                                   output,
                                   block_matched_field,
                                   block_stencil,
                                   matching_stencil,
                                   threshold,
                                   max_block_matches,
                                   compilation_target,
                                   **compilation_kwargs)
