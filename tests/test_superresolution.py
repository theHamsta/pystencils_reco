# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from os.path import dirname, join

import numpy as np
import pytest
import skimage.io
import sympy

import pystencils
import pystencils_reco.transforms
from pystencils.data_types import create_type
from pystencils_reco import crazy
from pystencils_reco._projective_matrix import ProjectiveMatrix
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


def test_torch_simple():

    import pytest
    pytest.importorskip("torch")
    import torch

    x, y = pystencils.fields('x,y: float32[2d]')

    @crazy
    def move(x, y):
        h = pystencils.fields('h(8): float32[2d]')
        A = sympy.Matrix([[h.center(0), h.center(1), h.center(2)],
                          [h.center(3), h.center(4), h.center(5)],
                          [h.center(6), h.center(7), 1]])
        return {
            y.center: x.interpolated_access(ProjectiveMatrix(A) @ pystencils.x_vector(2))

        }

    kernel = move(x, y).create_pytorch_op()
    pystencils.autodiff.show_code(kernel.ast)

    x = torch.ones((10, 40)).cuda()
    h = torch.ones((10, 40, 8)).cuda()

    y = kernel().forward(h, x)

    # with autograd
    x = torch.ones((10, 40), requires_grad=True).cuda()
    h = torch.ones((10, 40, 8), requires_grad=True).cuda()

    y = kernel().forward(h, x)[0]

    assert y.requires_grad

    loss = y.mean()
    loss.backward()
    # kernel().forward(*([1]*9), x, y)


def test_torch_matrix():

    import pytest
    pytest.importorskip("torch")
    import torch

    # x, y = torch.zeros((20, 20)), torch.zeros((20, 20))
    x, y = pystencils.fields('x,y: float32[2d]')
    a = sympy.Symbol('a')

    @crazy
    def move(x, y, a):
        return {
            y.center: x.interpolated_access((pystencils.x_, pystencils.y_ + a))
        }

    kernel = move(x, y, a).create_pytorch_op()
    pystencils.autodiff.show_code(kernel.ast)


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
            return sympy.Matrix((args.norm(),
                                 sympy.atan2(args[1] - x.shape[1] / 2,
                                             args[0] - x.shape[0] / 2) / sympy.pi * x.shape[1] / 2))

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
                (args.norm(),
                 sympy.atan2(args[1] - x.shape[1] / 2,
                             args[0] - x.shape[0] / 2) / sympy.pi * x.shape[1] / 2))

        def inv():
            return lambda l: (sympy.Matrix((sympy.cos(l[1] * sympy.pi / x.shape[1] * 2) * l[0],
                                            sympy.sin(l[1] * sympy.pi / x.shape[1] * 2) * l[0]))
                              + sympy.Matrix(x.shape) * 0.5)

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


def test_spatial_derivative():
    x, y = pystencils.fields('x, y:  float32[2d]')
    tx, ty = pystencils.fields('t_x, t_y: float32[2d]')

    assignments = pystencils.AssignmentCollection({
        y.center: x.interpolated_access((tx.center + pystencils.x_, 2 * ty.center + pystencils.y_))
    })

    backward_assignments = pystencils.autodiff.create_backward_assignments(assignments)

    print("assignments: " + str(assignments))
    print("backward_assignments: " + str(backward_assignments))


def test_spatial_derivative2():
    import pystencils.interpolation_astnodes
    x, y = pystencils.fields('x, y:  float32[2d]')
    tx, ty = pystencils.fields('t_x, t_y: float32[2d]')

    assignments = pystencils.AssignmentCollection({
        y.center: x.interpolated_access((tx.center + pystencils.x_, ty.center + 2 * pystencils.y_))
    })

    backward_assignments = pystencils.autodiff.create_backward_assignments(assignments)

    assert backward_assignments.atoms(pystencils.interpolation_astnodes.DiffInterpolatorAccess)

    print("assignments: " + str(assignments))
    print("backward_assignments: " + str(backward_assignments))


def test_get_shift():
    from pystencils_autodiff.framework_integration.datahandling import PyTorchDataHandling

    lenna_file = join(dirname(__file__), "test_data", "lenna.png")
    lenna = skimage.io.imread(lenna_file, as_gray=True).astype(np.float32)

    dh = PyTorchDataHandling(lenna.shape)
    x, y, tx, ty = dh.add_arrays('x, y, tx, ty')

    dh.cpu_arrays['x'] = lenna
    dh.cpu_arrays['tx'][...] = 0.7
    dh.cpu_arrays['ty'][...] = -0.7
    dh.all_to_gpu()

    kernel = pystencils_reco.AssignmentCollection({
        y.center: x.interpolated_access((tx.center + pystencils.x_, 2 * ty.center +
                                         pystencils.y_), interpolation_mode='cubic_spline')
    }).create_pytorch_op()().forward

    dh.run_kernel(kernel)

    pyconrad.imshow(dh.gpu_arrays)


@pytest.mark.parametrize('scalar_experiment', (False,))
def test_get_shift_tensors(scalar_experiment):
    from pystencils_autodiff.framework_integration.datahandling import PyTorchDataHandling
    import torch

    lenna_file = join(dirname(__file__), "test_data", "lenna.png")
    lenna = skimage.io.imread(lenna_file, as_gray=True).astype(np.float32)

    dh = PyTorchDataHandling(lenna.shape)
    x, y, tx, ty = dh.add_arrays('xw, yw, txw, tyw')

    dh.cpu_arrays['xw'] = lenna
    dh.cpu_arrays['txw'][...] = 0.7
    dh.cpu_arrays['tyw'][...] = -0.7
    dh.all_to_gpu()
    pyconrad.imshow(dh.gpu_arrays)

    kernel = pystencils_reco.AssignmentCollection({
        y.center: x.interpolated_access((tx.center + pystencils.x_, 2 * ty.center + pystencils.y_))
    }).create_pytorch_op()().call

    y_array = dh.run_kernel(kernel)

    dh = PyTorchDataHandling(lenna.shape)
    x, y, tx, ty = dh.add_arrays('x, y, tx, ty')

    if scalar_experiment:
        var_x = torch.zeros((), requires_grad=True)
        var_y = torch.zeros((), requires_grad=True)
    else:
        var_x = torch.zeros(lenna.shape, requires_grad=True)
        var_y = torch.zeros(lenna.shape, requires_grad=True)

    dh.cpu_arrays.x = lenna

    assignments = pystencils_reco.AssignmentCollection({
        y.center: x.interpolated_access((tx.center + pystencils.x_,
                                         2 * ty.center + pystencils.y_))
    })

    print(pystencils.autodiff.create_backward_assignments(assignments))
    kernel = assignments.create_pytorch_op()

    print(kernel.ast)
    kernel = kernel().call

    learning_rate = 0.1
    params = (var_x, var_y)
    # assert all([p.is_leaf for p in params])
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    for i in range(100):
        if scalar_experiment:
            dh.cpu_arrays.tx = torch.ones(lenna.shape) * var_x
            dh.cpu_arrays.ty = torch.ones(lenna.shape) * var_y
        else:
            dh.cpu_arrays.tx = var_x
            dh.cpu_arrays.ty = var_y
        dh.all_to_gpu()

        y = dh.run_kernel(kernel)
        loss = (y - y_array).norm()

        optimizer.zero_grad()

        loss.backward(retain_graph=True)
        assert y.requires_grad

        optimizer.step()
        print(loss.cpu().detach().numpy())
        print("var_x: " + str(var_x.mean()))
        pyconrad.imshow(var_x)
    # pyconrad.imshow(dh.gpu_arrays)
    pyconrad.imshow(dh.gpu_arrays, wait_window_close=True)


@pytest.mark.parametrize('with_spline', ('with_spline', False))
def test_spline_diff(with_spline):
    from pystencils.fd import Diff
    from pystencils.datahandling import SerialDataHandling

    lenna_file = join(dirname(__file__), "test_data", "lenna.png")
    lenna = skimage.io.imread(lenna_file, as_gray=True).astype(np.float32)

    dh = SerialDataHandling(lenna.shape, default_target='gpu', default_ghost_layers=0, default_layout='numpy')
    x, y, tx, ty = dh.add_arrays('x, y, tx, ty', dtype=np.float32)

    dh.cpu_arrays['x'] = lenna
    dh.cpu_arrays['tx'][...] = 0.7
    dh.cpu_arrays['ty'][...] = -0.7
    out = dh.add_array('out', dtype=np.float32)
    dh.all_to_gpu()

    kernel = pystencils_reco.AssignmentCollection({
        y.center: Diff(x, 0).interpolated_access((tx.center + pystencils.x_,
                                                  ty.center + pystencils.y_),
                                                 interpolation_mode='cubic_spline' if with_spline else 'linear')
    }).compile(target='gpu')

    dh.run_kernel(kernel)

    print(pystencils.show_code(kernel))

    kernel = pystencils_reco.AssignmentCollection({
        out.center: x.interpolated_access((tx.center + pystencils.x_, ty.center + pystencils.y_),
                                          interpolation_mode='cubic_spline' if with_spline else 'linear')
    }).compile(target='gpu')

    dh.run_kernel(kernel)

    print(pystencils.show_code(kernel))

    pyconrad.imshow(dh.gpu_arrays)
    pyconrad.imshow(dh.gpu_arrays)


@pytest.mark.parametrize('scalar_experiment', (False,))
def test_rotation(scalar_experiment):
    from pystencils_autodiff.framework_integration.datahandling import PyTorchDataHandling
    from pystencils_reco.resampling import rotation_transform

    import torch

    lenna_file = join(dirname(__file__), "test_data", "lenna.png")
    lenna = skimage.io.imread(lenna_file, as_gray=True).astype(np.float32)

    GROUNDTRUTH_ANGLE = 0.3

    target = np.zeros(lenna.shape)
    rotation_transform(lenna, target, GROUNDTRUTH_ANGLE)()
    target = torch.Tensor(target).cuda()

    dh = PyTorchDataHandling(lenna.shape)
    x, y, angle = dh.add_arrays('x, y, angle')

    if scalar_experiment:
        var_angle = torch.zeros((), requires_grad=True)
    else:
        var_angle = torch.zeros(lenna.shape, requires_grad=True)

    var_lenna = torch.autograd.Variable(torch.from_numpy(
        lenna + np.random.randn(*lenna.shape).astype(np.float32)), requires_grad=True)
    assert var_lenna.requires_grad

    learning_rate = 0.1
    params = (var_angle, var_lenna)

    optimizer = torch.optim.Adam(params, lr=learning_rate)

    assignments = rotation_transform(x, y, angle)
    kernel = assignments.create_pytorch_op()
    print(kernel)

    kernel = kernel().call

    for i in range(100000):
        if scalar_experiment:
            dh.cpu_arrays.angle = torch.ones(lenna.shape) * (var_angle + 0.29)
        else:
            dh.cpu_arrays.angle = var_angle
        dh.cpu_arrays.x = var_lenna
        dh.all_to_gpu()

        y = dh.run_kernel(kernel)
        loss = (y - target).norm()

        optimizer.zero_grad()

        loss.backward(retain_graph=True)
        assert y.requires_grad

        optimizer.step()
        print(loss.cpu().detach().numpy())
        pyconrad.imshow(var_lenna)

    pyconrad.show_everything()
