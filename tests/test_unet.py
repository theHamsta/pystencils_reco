# -*- coding: utf-8 -*-
#
# Copyright © 2019 Tom Harke, Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import numpy as np

import pystencils as ps
from pystencils_reco.unet import max_pooling, relu


def test_unet():
    pass


def test_conv():
    src_arr = np.random.rand(21, 31, 1)
    dst_arr = np.zeros([21, 31, 3])
    stencil_arr = np.ones([3, 3, 1, 3]) / 9

    dst, src = ps.fields(f'dst(3), src(1): [21, 31]')

    stencil = ps.fields(f'stencil(1, 3): [3,3]')
    stencil.field_type = ps.field.FieldType.CUSTOM

    assignments = conv_2d_stencil(src, stencil, dst)
    ast = ps.create_kernel(assignments)

    print(ps.show_code(ast))
    kernel = ast.compile()

    kernel(dst=dst_arr, src=src_arr, stencil=stencil_arr)


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


test_max_pooling()
