# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import os

import pytest

import pystencils
import pystencils_reco.resampling
import sympy
from pystencils.autodiff import torch_tensor_from_field
from pystencils_reco.filters import mean_filter
from pystencils_reco.projection import forward_projection
from pystencils_reco.stencils import BallStencil

if 'CI' in os.environ:
    pytest.skip('torch destroys pycuda tests',  allow_module_level=True)


def test_pytorch():
    block_stencil = BallStencil(1, ndim=2)

    x, y = pystencils.fields('x,y: float32[100,100]')
    filter = mean_filter(x, y, block_stencil)
    print(filter)
    print(filter.backward())

    x_tensor = torch_tensor_from_field(x, requires_grad=True, cuda=False)
    y_tensor = torch_tensor_from_field(y, cuda=False)

    torch_op = filter.create_pytorch_op(x=x_tensor + 1, y=y_tensor)
    print(torch_op)


def test_pytorch_gpu():
    block_stencil = BallStencil(1, ndim=2)

    x, y = pystencils.fields('x,y: float32[100,100]')
    filter = mean_filter(x, y, block_stencil)
    print(filter)
    print(filter.backward())

    x_tensor = torch_tensor_from_field(x, requires_grad=True, cuda=True)
    y_tensor = torch_tensor_from_field(y, cuda=True)

    torch_op = filter.create_pytorch_op(x=x_tensor + 1, y=y_tensor)
    print(torch_op)


def test_pytorch_from_tensors():
    block_stencil = BallStencil(1, ndim=2)

    x, y = pystencils.fields('x,y: float32[100,100]')
    x_tensor = torch_tensor_from_field(x, requires_grad=True, cuda=False)
    y_tensor = torch_tensor_from_field(y, cuda=False)
    torch_op = mean_filter(x_tensor, y_tensor, block_stencil)

    print(torch_op)


@pytest.mark.skip(reason="native texture uploading not implemented")
def test_texture():
    x, y = pystencils.fields('x,y: float32[100,100]')
    assignments = pystencils_reco.resampling.scale_transform(x, y, 2)

    x_tensor = torch_tensor_from_field(x, requires_grad=True, cuda=True)
    y_tensor = torch_tensor_from_field(y, cuda=True)
    kernel = assignments.create_pytorch_op(x=x_tensor, y=y_tensor)
    print(assignments)
    print(kernel)
    kernel.forward()


@pytest.mark.skip(reason="native texture uploading not implemented")
def test_projection():

    volume = pystencils.fields('volume: float32[100,200,300]')
    projections = pystencils.fields('projections: float32[600,500]')

    projection_matrix = sympy.Matrix([[-289.0098977737411, -1205.2274801832275, 0.0, 186000.0],
                                      [-239.9634468375339, - 4.188577544948043, 1200.0, 144000.0],
                                      [-0.9998476951563913, -0.01745240643728351, 0.0, 600.0]])

    assignments = forward_projection(volume, projections, projection_matrix)

    x_tensor = torch_tensor_from_field(volume, requires_grad=True, cuda=True)
    y_tensor = torch_tensor_from_field(projections, cuda=True)
    kernel = assignments.create_pytorch_op(volume=x_tensor, projections=y_tensor)
    print(assignments)
    print(kernel)
