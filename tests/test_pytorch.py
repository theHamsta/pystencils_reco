# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import torch

import pystencils
from pystencils.autodiff import torch_tensor_from_field
from pystencils_reco.filters import mean_filter
from pystencils_reco.stencils import BallStencil


def test_pytorch():
    block_stencil = BallStencil(1, ndim=2)

    x, y = pystencils.fields('x,y: float32[100,100]')
    filter = mean_filter(x, y, block_stencil)
    print(filter)
    print(filter.backward())

    x_tensor = torch_tensor_from_field(x, requires_grad=True, cuda=False)
    y_tensor = torch_tensor_from_field(y, cuda=False)

    torch_op = filter.create_pytorch_op(x=x_tensor+1, y=y_tensor)
    print(torch_op)


def test_pytorch_gpu():
    block_stencil = BallStencil(1, ndim=2)

    x, y = pystencils.fields('x,y: float32[100,100]')
    filter = mean_filter(x, y, block_stencil)
    print(filter)
    print(filter.backward())

    x_tensor = torch_tensor_from_field(x, requires_grad=True, cuda=True)
    y_tensor = torch_tensor_from_field(y, cuda=True)

    torch_op = filter.create_pytorch_op(x=x_tensor+1, y=y_tensor)
    print(torch_op)


def test_pytorch_from_tensors():
    block_stencil = BallStencil(1, ndim=2)

    x, y = pystencils.fields('x,y: float32[100,100]')
    x_tensor = torch_tensor_from_field(x, requires_grad=True, cuda=True)
    y_tensor = torch_tensor_from_field(y, cuda=True)
    filter = mean_filter(x_tensor, y_tensor, block_stencil)
    print(filter)
    print(filter.backward())

    torch_op = filter.create_pytorch_op(x=x_tensor+1, y=y_tensor)
    print(torch_op)


def main():
    # test_pytorch()
    test_pytorch()


if __name__ == '__main__':
    main()
