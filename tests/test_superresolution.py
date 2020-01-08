# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from os.path import dirname, join
from time import sleep

import numpy as np
import skimage.io
import sympy

import pystencils
import pystencils_reco.transforms
from pystencils_reco.filters import gauss_filter
from pystencils_reco.resampling import (
    downsample, resample, resample_to_shape, scale_transform, translate)
from pystencils_reco.stencils import BallStencil

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

    lenna_file = join(dirname(__file__), "test_data", "lenna.png")
    lenna = skimage.io.imread(lenna_file, as_gray=True).astype(np.float32)

    transformed = np.zeros((400, 400), np.float32)

    resample(x, y).compile()(x=lenna, y=transformed)

    pyconrad.show_everything()
    # while True:
    # sleep(100)


def test_polar_transform2():
    x, y = pystencils.fields('x, y:  float32[2d]')

    class PolarTransform(sympy.Function):
        def eval(args):
            return sympy.Matrix(
                (args.norm(), sympy.atan2(args[1]-x.shape[1]/2, args[0]-x.shape[0]/2) / sympy.pi * x.shape[1]/2))

    x.set_coordinate_origin_to_field_center()
    y.coordinate_transform = PolarTransform
    y.set_coordinate_origin_to_field_center()

    lenna_file = join(dirname(__file__), "test_data", "lenna.png")
    lenna = skimage.io.imread(lenna_file, as_gray=True).astype(np.float32)

    transformed = np.zeros((400, 400), np.float32)

    resample(x, y).compile()(x=lenna, y=transformed)

    pyconrad.show_everything()
    # while True:
    # sleep(100)


def test_polar_inverted_transform():
    x, y = pystencils.fields('x, y:  float32[2d]')

    class PolarTransform(sympy.Function):
        def eval(args):
            return sympy.Matrix(
                (args.norm(), sympy.atan2(args[1]-x.shape[1]/2, args[0]-x.shape[0]/2) / sympy.pi * x.shape[1]/2))

        def inv():
            return lambda l: sympy.Matrix((sympy.cos(l[1] * sympy.pi / x.shape[1]*2) * l[0],
                                           sympy.sin(l[1] * sympy.pi / x.shape[1]*2) * l[0])) + sympy.Matrix(x.shape) * 0.5

    lenna_file = join(dirname(__file__), "test_data", "lenna.png")
    lenna = skimage.io.imread(lenna_file, as_gray=True).astype(np.float32)

    # transformed = np.zeros((400, 400), np.float32)
    # back_transformed = np.zeros((400, 400), np.float32)
    transformed = np.zeros_like(lenna)
    back_transformed = np.zeros_like(lenna)

    x.set_coordinate_origin_to_field_center()
    y.coordinate_transform = PolarTransform
    y.set_coordinate_origin_to_field_center()
    resample(x, y).compile()(x=lenna, y=transformed)
    resample(y, x).compile()(x=back_transformed, y=transformed)

    pyconrad.show_everything()
    # while True:
    # sleep(100)


def test_shift():
    x, y = pystencils.fields('x, y:  float32[2d]')

    class ShiftTransform(sympy.Function):
        def eval(args):
            return args + sympy.Matrix((5, 5))

        def inv():
            return lambda l: l - sympy.Matrix((5, 5))

    lenna_file = join(dirname(__file__), "test_data", "lenna.png")
    lenna = skimage.io.imread(lenna_file, as_gray=True).astype(np.float32)

    transformed = np.zeros_like(lenna)
    back_transformed = np.zeros_like(lenna)

    x.set_coordinate_origin_to_field_center()
    y.coordinate_transform = ShiftTransform
    y.set_coordinate_origin_to_field_center()
    resample(x, y).compile()(x=lenna, y=transformed)
    resample(y, x).compile()(x=back_transformed, y=transformed)

    diff = lenna - back_transformed
    assert diff is not None

    pyconrad.show_everything()
    # while True:
    # sleep(100)


def test_motion_model():
    x, y = pystencils.fields('x, y:  float32[2d]')
    transform_field = pystencils.fields('t_x, t_y: float32[2d]')

    lenna_file = join(dirname(__file__), "test_data", "lenna.png")
    lenna = skimage.io.imread(lenna_file, as_gray=True).astype(np.float32)

    # transformed = np.zeros((400, 400), np.float32)
    # back_transformed = np.zeros((400, 400), np.float32)
    transformed = np.zeros_like(lenna)
    back_transformed = np.zeros_like(lenna)
    translate_x = np.zeros((10, 10), np.float32)
    translate_y = np.zeros((10, 10), np.float32)

    pystencils_reco.transforms.extend_to_size_of_other_field(transform_field[0], x)
    pystencils_reco.transforms.extend_to_size_of_other_field(transform_field[1], x)

    shift = sympy.Matrix(sympy.symbols('s:2'))
    shift_val = sympy.Matrix([transform_field[i].interpolated_access(
        transform_field[i].physical_to_index(x.physical_coordinates))
        for i in range(x.ndim)])

    class ShiftTransform(sympy.Function):
        def eval(args):
            return args + shift

        def inv():
            return lambda args: args - shift

    y.coordinate_transform = ShiftTransform
    pystencils_reco.AssignmentCollection([*resample(x, y),
                                          *[pystencils.Assignment(shift[i], shift_val[i]) for i in range(2)]]
                                         ).compile()(x=lenna, y=transformed, t_x=translate_x, t_y=translate_y)

    pystencils_reco.AssignmentCollection([*resample(x, y),
                                          *[pystencils.Assignment(shift[i], shift_val[i]) for i in range(2)]]
                                         ).compile()(x=back_transformed,
                                                     y=transformed,
                                                     t_x=translate_x,
                                                     t_y=translate_y)

    pyconrad.show_everything()
    # while True:
    # sleep(100)


def test_motion_model2():
    x, y = pystencils.fields('x, y:  float32[2d]')
    transform_field = pystencils.fields('t_x, t_y: float32[2d]')

    lenna_file = join(dirname(__file__), "test_data", "lenna.png")
    lenna = skimage.io.imread(lenna_file, as_gray=True).astype(np.float32)

    # transformed = np.zeros((400, 400), np.float32)
    # back_transformed = np.zeros((400, 400), np.float32)
    transformed = np.zeros_like(lenna)
    blurred = np.zeros_like(lenna)

    translate_x = np.zeros_like(lenna)
    translate_y = np.zeros_like(lenna)
    amplitude = 20

    resample_to_shape(amplitude * np.random.randn(10, 10).astype(np.float32), lenna.shape).compile()(output=translate_x)
    resample_to_shape(amplitude * np.random.randn(10, 10).astype(np.float32), lenna.shape).compile()(output=translate_y)

    translate(x, y, sympy.Matrix((transform_field[0].center, transform_field[1].center))
              ).compile()(x=lenna, y=transformed, t_x=translate_x, t_y=translate_y)

    # resample(x, y).compile()(x=back_transformed, y=transformed, t_x=translate_x, t_y=translate_y)

    kernel = gauss_filter(transformed, blurred, BallStencil(5, ndim=2), 10).compile()
    print(pystencils.show_code(kernel))
    kernel(input_field=transformed, output_field=blurred)

    pyconrad.show_everything()

    # while True:
    # sleep(100)
