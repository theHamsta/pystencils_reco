# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from os.path import dirname, join

import numpy as np
import skimage.io
import sympy

import pystencils
from pystencils_reco.resampling import downsample, resample, scale_transform, translate

try:
    import pyconrad.autoinit
except Exception:
    import unittest.mock
    pyconrad = unittest.mock.MagicMock()


def test_superresolution():

    x, y = np.random.rand(20, 10), np.zeros((20, 10))

    kernel = scale_transform(x, y, 0.5).compile()
    print(pystencils.show_code(kernel))
    kernel()

    pyconrad.show_everything()


def test_downsample():
    shape = (20, 10)

    x, y = np.random.rand(*shape), np.zeros(tuple(s // 2 for s in shape))

    kernel = downsample(x, y, 2).compile()
    print(pystencils.show_code(kernel))
    kernel()

    pyconrad.show_everything()


def test_warp():
    import torch
    NUM_LENNAS = 5
    perturbation = 0.1

    lenna_file = join(dirname(__file__), "test_data", "lenna.png")
    lenna = skimage.io.imread(lenna_file, as_gray=True).astype(np.float32)

    warp_vectors = list(perturbation * torch.randn(lenna.shape + (2,)) for _ in range(NUM_LENNAS))

    warped = [torch.zeros(lenna.shape) for _ in range(NUM_LENNAS)]

    warp_kernel = translate(lenna, warped[0], pystencils.autodiff.ArrayWrapper(
        warp_vectors[0], index_dimensions=1), interpolation_mode='linear').compile()

    for i in range(len(warped)):
        warp_kernel(lenna[i], warped[i], warp_vectors[i])


def test_polar_transform():
    x, y = pystencils.fields('x, y:  float32[2d]')

    x.set_coordinate_origin_to_field_center()
    y.set_coordinate_origin_to_field_center()
    y.coordinate_transform = lambda x: sympy.Matrix((x.norm(), sympy.atan2(*x) / (2 * sympy.pi) * y.shape[1]))

    lenna_file = join(dirname(__file__), "test_data", "lenna.png")
    lenna = skimage.io.imread(lenna_file, as_gray=True).astype(np.float32)

    transformed = np.zeros((500, 500), np.float32)

    resample(x, y).compile()(x=lenna, y=transformed)

    pyconrad.imshow(transformed)
