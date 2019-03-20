# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pystencils
from pystencils.autodiff import torch_tensor_from_field
from pystencils_reco.block_matching import block_matching_integer_offsets
from pystencils_reco.stencils import BallStencil


def test_pytorch():
    block_stencil = BallStencil(1, ndim=2)
    #
    matching_stencil = BallStencil(1, ndim=2)

    x, y, matches = pystencils.fields('x,y, matches(%i): float32[100,100]' % len(matching_stencil))
    block_matching = block_matching_integer_offsets(x, y, matches, block_stencil, matching_stencil)
    print(block_matching)
    print(block_matching.backward())

    x_tensor = torch_tensor_from_field(x, requires_grad=True)
    y_tensor = torch_tensor_from_field(y)
    matches_tensor = torch_tensor_from_field(matches)

    torch_op = block_matching.create_pytorch_op(x=x_tensor+1, y=y_tensor, matches=matches_tensor)
    print(torch_op)


def main():
    test_pytorch()


if __name__ == '__main__':
    main()
