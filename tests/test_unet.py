# -*- coding: utf-8 -*-
#
# Copyright © 2019 Tom Harke, Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import numpy as np
import pytest

import pystencils as ps
from pystencils_reco.unet import channel_convolution, convolution, max_pooling, relu


def test_unet():
    pass


def test_relu():
    src_arr = np.random.rand(20, 30, 4) - 0.5
    dst_arr = np.zeros([20, 30, 4])

    dst, src = ps.fields('dst(4), src(4): [2d]', dst=dst_arr, src=src_arr)

    assignments = relu(src, dst)
    kernel = ps.create_kernel(assignments)
    compiled = kernel.compile()
    compiled(src=src_arr, dst=dst_arr)


def test_max_pooling():
    src_arr = np.random.rand(20, 40, 4)
    dst_arr = np.zeros([10, 20, 4])

    # only source needs to be CUSTOM

    assignments = max_pooling(src_arr, dst_arr)
    compiled = assignments.compile()

    compiled(src=src_arr, dst=dst_arr)


@pytest.mark.parametrize('input_channels, output_channels', ((1, 1), (3, 2), (2, 4)))
def test_conv(input_channels, output_channels):
    src_arr = np.random.rand(21, 31, input_channels)
    dst_arr = np.zeros([21, 31, output_channels])
    stencil_arr = np.ones([3, 3, input_channels, output_channels]) / 9

    dst, src = ps.fields(f'dst({output_channels}), src({input_channels}): [2d]')

    stencil = ps.fields(f'stencil({input_channels}, {output_channels}): [3,3]')
    stencil.field_type = ps.field.FieldType.CUSTOM

    assignments = channel_convolution(src, stencil, dst)
    ast = ps.create_kernel(assignments)

    print(ps.show_code(ast))

    kernel = ast.compile()

    kernel(dst=dst_arr, src=src_arr, stencil=stencil_arr)


@pytest.mark.parametrize('input_channels, output_channels', ((1, 1), (3, 2), (2, 4)))
def test_conv_advanced(input_channels, output_channels):
    filter_shape = (5, 4)

    src_arr = np.random.rand(21, 31, input_channels)
    dst_arr = np.zeros([21, 31, output_channels])
    stencil_arr = np.ones([*filter_shape, input_channels, output_channels]) / (5 * 4)

    dst, src = ps.fields(f'dst({output_channels}), src({input_channels}): [2d]')

    stencil = ps.fields(f'stencil({input_channels}, {output_channels}): [{filter_shape[0]}, {filter_shape[1]}]')
    stencil.field_type = ps.field.FieldType.CUSTOM

    assignments = channel_convolution(src, stencil, dst)
    ast = ps.create_kernel(assignments)

    print(ps.show_code(ast))
    kernel = ast.compile()
    kernel(dst=dst_arr, src=src_arr, stencil=stencil_arr)


@pytest.mark.parametrize('input_channels, output_channels', ((1, 1), (3, 2), (2, 4)))
def test_conv_3d(input_channels, output_channels):
    filter_shape = (5, 4, 3)

    src_arr = np.random.rand(21, 31, 42, input_channels)
    dst_arr = np.zeros([21, 31, 42, output_channels])
    stencil_arr = np.ones([*filter_shape, input_channels, output_channels]) / (5 * 4 * 3)

    channel_convolution(src_arr, stencil_arr, dst_arr).compile()()


def test_conv3d_without_channels():
    filter_shape = (5, 4, 3)

    src_arr = np.random.rand(21, 31, 42)
    dst_arr = np.zeros([21, 31, 42])
    stencil_arr = np.ones(filter_shape) / (5 * 4 * 3)

    convolution(src_arr, stencil_arr, dst_arr).compile()()
