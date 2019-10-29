# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import os

import numpy as np
import pytest
import sympy

import pystencils
import pystencils_reco.resampling
from pystencils_autodiff.field_tensor_conversion import torch_tensor_from_field
from pystencils_reco import crazy
from pystencils_reco.filters import mean_filter
from pystencils_reco.projection import forward_projection
from pystencils_reco.stencils import BallStencil, BoxStencil

# if 'PYTORCH_TEST' not in os.environ:
# pytest.skip('torch destroys pycuda tests',  allow_module_level=True)


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


def test_texture():
    x, y = pystencils.fields('x,y: float32[100,100]')
    assignments = pystencils_reco.resampling.scale_transform(x, y, 2)

    x_tensor = torch_tensor_from_field(x, requires_grad=True, cuda=True)
    y_tensor = torch_tensor_from_field(y, cuda=True)
    kernel = assignments.create_pytorch_op(x=x_tensor, y=y_tensor)
    print(assignments)
    print(kernel)
    kernel()


@pytest.mark.parametrize('ndim', (3,))
def test_texture_crazy(ndim):
    import torch
    x = torch.rand((20, 30, 100)[0:ndim])
    y = torch.empty_like(x)
    scale = sympy.Symbol('scale')

    kernel = pystencils_reco.resampling.scale_transform(x, y, scale).compile(target='gpu')
    print(kernel.code)
    rtn = kernel().forward(input_field=x, output_field=y, transform_matrix=2)[0]
    import pyconrad.autoinit
    pyconrad.imshow(x)
    pyconrad.imshow(rtn)


def test_numpy_crazy():
    x = np.random.rand(20, 30)
    y = np.empty_like(x)

    @crazy
    def kernel(x, y):
        return pystencils_reco.AssignmentCollection({y.center: x.center + 1})

    kernel = kernel(x, y).compile()
    kernel()
    import pyconrad.autoinit
    pyconrad.imshow(x, 'x')
    pyconrad.imshow(y, 'y')


def test_torch_crazy():
    import torch
    x = torch.rand((20, 30)).cuda()
    y = torch.empty_like(x).cuda()
    print("y.dtype: " + str(y.dtype))

    @crazy
    def kernel(x, y):
        return pystencils_reco.AssignmentCollection({y.center: x.center + 1})

    assignments = kernel(x, y)
    print("assignments: " + str(assignments))
    kernel = assignments.compile()()
    print("kernel.code: " + str(kernel.code))
    rtn = kernel.forward(x=x, y=y)[0]
    assert rtn is y
    import pyconrad.autoinit
    pyconrad.imshow(x)
    pyconrad.imshow(rtn)
    pyconrad.imshow(y)


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


def test_mean_filter_with_crazy_torch():
    torch = pytest.importorskip('torch')
    x = torch.rand((20, 30))
    y = torch.zeros_like(x)

    assignments = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=2))
    ab = assignments.create_pytorch_op()
    ab()
    assignments.compile()()


test_texture_crazy(3)
