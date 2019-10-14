# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from os.path import dirname, join

import numpy as np
import pytest
import skimage.io

import pystencils
from pystencils_reco.resampling import rotation_transform, scale_transform

try:
    import pyconrad.autoinit
except Exception:
    from unittest.mock import MagicMock
    pyconrad = MagicMock()


def test_scaling():

    for ndim in range(1, 5):
        for scale in (0.5, [(s + 1) * 0.1 for s in range(ndim)]):
            x, y = pystencils.fields('x,y: float32[%id]' % ndim)
            transform = scale_transform(x, y, scale)
            print(transform)


def test_rotation():

    for ndim in (2, 3):
        for angle in (0.4, -0.8):
            for axis in range(3):
                x, y = pystencils.fields('x,y: float32[%id]' % ndim)
                transform = rotation_transform(x, y, angle, axis)
                print(transform)


def test_scaling_compilation():

    for ndim in range(1, 4):
        for scale in (0.5, [(s + 1) * 0.1 for s in range(ndim)]):
            x, y = pystencils.fields('x,y: float32[%id]' % ndim)
            scale_transform(x, y, scale).compile('gpu')


def test_rotation_compilation():

    for ndim in (2, 3):
        for angle in (0.4, -0.8):
            for axis in range(3):
                x, y = pystencils.fields('x,y: float32[%id]' % ndim)
                rotation_transform(x, y, angle, axis).compile('gpu')


def test_scaling_visualize():
    from pycuda.gpuarray import to_gpu, zeros_like

    x, y = pystencils.fields('x,y: float32[2d]')
    s = pystencils.data_types.TypedSymbol('s', 'float32')
    transform = scale_transform(x, y, s).compile('gpu')

    test_image = 1 - skimage.io.imread(join(dirname(__file__), "test_data", "test_vessel2d_mask.png"), as_gray=True)
    test_image = np.ascontiguousarray(test_image, np.float32)
    test_image = to_gpu(test_image)
    tmp = zeros_like(test_image)

    for s in (0.2, 0.5, 0.7, 1, 2):
        transform(x=test_image, y=tmp, s=s)
        pyconrad.imshow(tmp.get(), str(s))


def test_rotation_visualize():
    from pycuda.gpuarray import to_gpu, zeros_like

    x, y = pystencils.fields('x,y: float32[2d]')
    s = pystencils.data_types.TypedSymbol('s', 'float32')
    transform = rotation_transform(x, y, s).compile('gpu')

    test_image = 1 - skimage.io.imread(join(dirname(__file__), "test_data", "test_vessel2d_mask.png"), as_gray=True)
    test_image = np.ascontiguousarray(test_image, np.float32)
    test_image = to_gpu(test_image)
    tmp = zeros_like(test_image)

    for s in (0, 0.2, 0.5, 0.7, 1, 2):
        transform(x=test_image, y=tmp, s=s)
        pyconrad.imshow(tmp.get(), str(s))


@pytest.mark.skip(reason="BSPLINE currently not in pystencils")
def test_rotation_visualize_bspline():
    from pycuda.gpuarray import to_gpu, zeros_like

    x, y = pystencils.fields('x,y: float32[2d]')
    s = pystencils.data_types.TypedSymbol('s', 'float32')
    transform = rotation_transform(x, y, s, interpolation_mode='cubic_spline').compile('gpu')

    test_image = 1 - skimage.io.imread(join(dirname(__file__), "test_data", "test_vessel2d_mask.png"), as_gray=True)
    test_image = np.ascontiguousarray(test_image, np.float32)
    test_image = to_gpu(test_image)
    pyconrad.imshow(test_image, "before")
    pystencils.gpucuda.texture_utils.prefilter_for_cubic_bspline(test_image)
    pyconrad.imshow(test_image, "prefiltered")
    tmp = zeros_like(test_image)

    for s in (0, 0.2, 0.5, 0.7, 1, 2):
        transform(x=test_image, y=tmp, s=s)
        pyconrad.imshow(tmp.get(), str(s) + " bispline")


def test_rotation_around_center_visualize():
    from pycuda.gpuarray import to_gpu, zeros_like

    test_image = 1 - skimage.io.imread(join(dirname(__file__), "test_data", "test_vessel2d_mask.png"), as_gray=True)
    test_image = np.ascontiguousarray(test_image, np.float32)
    test_image = to_gpu(test_image)

    tmp = zeros_like(test_image)
    x, y = pystencils.fields(f'x,y: float32[{",".join(str(i) for i in test_image.shape)}]')
    x.set_coordinate_origin_to_field_center()
    y.set_coordinate_origin_to_field_center()
    print(x.coordinate_origin)
    s = pystencils.data_types.TypedSymbol('s', 'float32')
    transform = rotation_transform(x, y, s).compile('gpu')

    for s in (0, 0.2, 0.5, 0.7, 1, 2):
        transform(x=test_image, y=tmp, s=s)
        pyconrad.imshow(tmp.get(), str(s))
