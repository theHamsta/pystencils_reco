# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pystencils
import pystencils_reco.resampling
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
    x_tensor = torch_tensor_from_field(x, requires_grad=True, cuda=False)
    y_tensor = torch_tensor_from_field(y, cuda=False)
    torch_op = mean_filter(x_tensor, y_tensor, block_stencil)

    print(torch_op)


def test_texture():

    x, y = pystencils.fields('x,y: float32[100,100]')
    assignments = pystencils_reco.resampling.scale_transform(x, y, 2)

    x_tensor = torch_tensor_from_field(x, requires_grad=True, cuda=True)
    y_tensor = torch_tensor_from_field(y, cuda=True)
    kernel = assignments.create_pytorch_op(x=x_tensor, y=y_tensor)
    print(assignments)
    print(kernel)


def main():
    # test_pytorch()
    # test_pytorch_from_tensors()
    test_texture()


if __name__ == '__main__':
    main()
