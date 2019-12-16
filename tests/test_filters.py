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
import pystencils.simp
import pystencils_reco.filters
from pystencils_reco.stencils import BoxStencil

try:
    # import pyconrad.autoinit
    raise ImportError()
except ImportError:  # NOQA
    from unittest.mock import MagicMock
    pyconrad = MagicMock()


def test_mean_filter():
    x, y = pystencils.fields('x,y: float32[2d]')

    assignments = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=2))
    print(assignments)

    assignments = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=2, with_center=False))
    print(assignments)


def test_mean_filter_with_crazy():
    x = np.random.rand(20, 30)
    y = np.zeros_like(x)

    assignments = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=2))
    print(assignments)

    assignments = pystencils_reco.filters.mean_filter(
        input_field=x, output_field=y, stencil=BoxStencil(3, ndim=2, with_center=False))
    print(assignments)


def test_mean_filter_with_crazy_compilation():
    x = np.random.rand(20, 30)
    y = np.zeros_like(x)

    assignments = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=2))
    assignments.compile('gpu')
    assignments.compile('cpu')(input_field=x, output_field=y)


def test_crazy_target_detection():
    to_gpu = pytest.importorskip('pycuda.gpuarray').to_gpu
    zeros_like = pytest.importorskip('pycuda.gpuarray').zeros_like
    x = to_gpu(np.random.rand(20, 30))
    y = zeros_like(x)

    assignments = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=2))
    assignments.compile()(input_field=x, output_field=y)


def test_mean_filter_evaluation():
    x, y = pystencils.fields('x,y: float32[2d]')
    x_array = np.random.rand(20, 23, 25).astype(np.float32)
    y_array = np.random.rand(20, 23, 25).astype(np.float32)

    assignments = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=2))
    ast = pystencils.create_kernel(assignments)
    kernel = ast.compile()
    kernel(x=x_array[0], y=y_array[0])

    kernel = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=2)).compile()
    kernel(x=x_array[0], y=y_array[0])

    kernel = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=2, with_center=False)).compile()
    kernel(x=x_array[0], y=y_array[0])

    x, y = pystencils.fields('x,y: float32[3d]')
    kernel = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=3)).compile()
    kernel(x=x_array, y=y_array)

    kernel = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=3, with_center=False)).compile()
    kernel(x=x_array, y=y_array)


@pytest.mark.skipif("CI" in os.environ and os.environ["CI"] == "true", reason="Skip GPU tests on CI")
def test_visualize_mean_filter():
    x, y = pystencils.fields('x,y: float32[2d]')
    x_array = np.random.rand(20, 23).astype(np.float32)
    y_array = np.random.rand(20, 23).astype(np.float32)

    kernel = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=2)).compile()
    kernel(x=x_array, y=y_array)

    pyconrad.imshow(x_array, 'x')
    pyconrad.imshow(y_array, 'y')


@pytest.mark.skipif("CI" in os.environ and os.environ["CI"] == "true", reason="Skip GPU tests on CI")
def test_mean_filter_evaluation_gpu():
    from pycuda.gpuarray import to_gpu
    x, y = pystencils.fields('x,y: float32[2d]')
    x_array = to_gpu(np.random.rand(20, 23, 25).astype(np.float32))
    y_array = to_gpu(np.random.rand(20, 23, 25).astype(np.float32))

    assignments = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=2))
    ast = pystencils.create_kernel(assignments, target='gpu')
    kernel = ast.compile()
    kernel(x=x_array[0], y=y_array[0])

    kernel = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=2)).compile(target='gpu')
    kernel(x=x_array[0], y=y_array[0])

    kernel = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=2, with_center=False)).compile(target='gpu')
    kernel(x=x_array[0], y=y_array[0])

    x, y = pystencils.fields('x,y: float32[3d]')
    kernel = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=3)).compile(target='gpu')
    kernel(x=x_array, y=y_array)

    kernel = pystencils_reco.filters.mean_filter(x, y, BoxStencil(3, ndim=3, with_center=False)).compile(target='gpu')
    kernel(x=x_array, y=y_array)


def test_gauss_filter():
    x, y = pystencils.fields('x,y: float32[2d]')

    assignments = pystencils_reco.filters.gauss_filter(x, y, BoxStencil(3, ndim=2), sigma=0.5)
    print(assignments)

    assignments = pystencils_reco.filters.gauss_filter(x, y, BoxStencil(3, ndim=2, with_center=False), sigma=0.5)
    print(assignments)

    assignments = pystencils_reco.filters.gauss_filter(
        x, y, BoxStencil(3, ndim=2, with_center=False), sigma=sympy.Symbol('a'))
    print(assignments)


def test_gauss_filter_evaluation():

    x_array = np.random.rand(20, 23, 25).astype(np.float32)
    y_array = np.random.rand(20, 23, 25).astype(np.float32)

    x, y = pystencils.fields('x,y: float32[3d]')
    kernel = pystencils_reco.filters.gauss_filter(x, y, BoxStencil(3, ndim=3), sigma=0.5).compile()
    kernel(x=x_array, y=y_array)

    x, y = pystencils.fields('x,y: float32[2d]')
    kernel = pystencils_reco.filters.gauss_filter(x, y, BoxStencil(3, ndim=2, with_center=False), sigma=0.5).compile()
    kernel(x=x_array[0], y=y_array[0])

    kernel = pystencils_reco.filters.gauss_filter(
        x, y, BoxStencil(3, ndim=2, with_center=False), sigma=sympy.Symbol('a')).compile()
    kernel(x=x_array[0], y=y_array[0], a=0.7)
