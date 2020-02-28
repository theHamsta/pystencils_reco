# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import hashlib
import pickle

import numpy as np
import pytest
import sympy

import pystencils
import pystencils_reco
from pystencils.data_types import create_type
from pystencils_reco.projection import forward_projection

try:
    import pyconrad.autoinit
except Exception:
    import unittest
    pyconrad = unittest.mock.MagicMock()

m0 = sympy.Matrix([[1, 0, 0],
                   [0, 0, 1]])
m1 = sympy.Matrix([[-289.0098977737411, -1205.2274801832275, 0.0, 186000.0],
                   [-239.9634468375339, - 4.188577544948043, 1200.0, 144000.0],
                   [-0.9998476951563913, -0.01745240643728351, 0.0, 600.0]])
m2 = sympy.Matrix([[1, 0, 0],
                   [0, 1, 0]])
m3 = sympy.Matrix([[0, 1, 0],
                   [0, 1, 1]])


_hash = hashlib.md5


def test_genereric_projection():
    volume = pystencils.fields('volume: float32[3d]')
    projections = pystencils.fields('projections: float32[2D]')

    projection_matrix = pystencils_reco.matrix_symbols('T', pystencils.data_types.create_type('float32'), 3, 4)

    assignments = forward_projection(volume, projections, projection_matrix)
    kernel = assignments.compile('gpu')
    pystencils.show_code(kernel)


def test_projection_cpu():
    volume = pystencils.fields('volume: float32[3d]')
    projections = pystencils.fields('projections: float32[2D]')

    projection_matrix = sympy.Matrix([[-289.0098977737411, -1205.2274801832275, 0.0, 186000.0],
                                      [-239.9634468375339, - 4.188577544948043, 1200.0, 144000.0],
                                      [-0.9998476951563913, -0.01745240643728351, 0.0, 600.0]])

    forward_projection(volume, projections, projection_matrix).compile()


def test_projection():
    volume = pystencils.fields('volume: float32[100,200,300]')
    projections = pystencils.fields('projections: float32[2D]')

    projection_matrix = sympy.Matrix([[-289.0098977737411, -1205.2274801832275, 0.0, 186000.0],
                                      [-239.9634468375339, - 4.188577544948043, 1200.0, 144000.0],
                                      [-0.9998476951563913, -0.01745240643728351, 0.0, 600.0]])

    assignments = forward_projection(volume, projections, projection_matrix)  # .compile(target='gpu')
    print(type(assignments))
    print(pickle.dumps(assignments))

    # a = sympy.Symbol('a')
    # projection_matrix = sympy.Matrix([[-289.0098977737411, -1205.2274801832275, 0.0, 186000.0],
    # [a, - 4.188577544948043, 1200.0, 144000.0],
    # [-0.9998476951563913, -0.01745240643728351, 0.0, 600.0]])

    # kernel = forward_projection(volume, projections, projection_matrix).compile(target='gpu')
    # print(kernel.code)

    a = sympy.symbols('a:12')
    a = [pystencils.TypedSymbol(s.name, 'float32') for s in a]
    A = sympy.Matrix(3, 4, lambda i, j: a[i * 4 + j])
    # A = sympy.MatrixSymbol('A', 3, 4)
    projection_matrix = A
    forward_projection(volume, projections, projection_matrix).compile(target='gpu')


@pytest.mark.parametrize('with_spline', (False,))
def test_project_shepp_logan(with_spline):
    from pycuda.gpuarray import to_gpu, GPUArray

    from sympy.matrices.dense import MutableDenseMatrix
    MutableDenseMatrix.__hash__ = lambda x: 1  # hash(tuple(x))
    try:
        import pyconrad.autoinit
        phantom3d = pyconrad.phantoms.shepp_logan(100, 100, 100)
        pyconrad.imshow(phantom3d, 'phantom')
    except Exception:
        phantom3d = np.random.rand(30, 31, 32)

    for i, projection_matrix in enumerate((m1,)):

        volume = pystencils.fields('volume: float32[100,100,100]')
        projections = pystencils.fields('projections: float32[1024,960]')
        volume.set_coordinate_origin_to_field_center()
        volume.coordinate_transform = sympy.rot_axis2(0.2)
        # volume.coordinate_transform = sympy.rot_axis3(0.1)
        volume.coordinate_transform = 3 * volume.coordinate_transform
        projections.set_coordinate_origin_to_field_center()

        kernel = forward_projection(volume,
                                    projections,
                                    projection_matrix,
                                    step_size=1,
                                    cubic_bspline_interpolation=with_spline)
        print(kernel)
        kernel = kernel.compile('gpu')
        # print(kernel.code)

        volume_gpu = to_gpu(np.ascontiguousarray(phantom3d, np.float32))
        if with_spline:
            pystencils.gpucuda.texture_utils.prefilter_for_cubic_bspline(volume_gpu)
        projection_gpu = GPUArray(projections.spatial_shape, np.float32)

        kernel(volume=volume_gpu, projections=projection_gpu)

        pyconrad.imshow(volume_gpu, 'volume ' + str(with_spline))
        pyconrad.imshow(projection_gpu, 'projections ' + str(i) + str(with_spline))

    for i, projection_matrix in enumerate((m1,)):
        angle = pystencils_reco.typed_symbols('angle', 'float32')

        volume = pystencils.fields('volume: float32[100,100,100]')
        projections = pystencils.fields('projections: float32[1024,960]')
        volume.set_coordinate_origin_to_field_center()
        volume.coordinate_transform = sympy.rot_axis2(angle)
        # volume.coordinate_transform = sympy.rot_axis3(0.1)
        volume.coordinate_transform = 3 * volume.coordinate_transform
        projections.set_coordinate_origin_to_field_center()

        kernel = forward_projection(volume,
                                    projections,
                                    projection_matrix,
                                    step_size=1,
                                    cubic_bspline_interpolation=with_spline)
        print(kernel)
        kernel = kernel.compile('gpu')
        # print(kernel.code)

        volume_gpu = to_gpu(np.ascontiguousarray(phantom3d, np.float32))
        if with_spline:
            pystencils.gpucuda.texture_utils.prefilter_for_cubic_bspline(volume_gpu)
        projection_gpu = GPUArray(projections.spatial_shape, np.float32)

        for phi in np.arange(0, np.pi, np.pi / 100):
            kernel(volume=volume_gpu, projections=projection_gpu, angle=phi)
            pyconrad.imshow(projection_gpu, 'rotation!' + str(with_spline))
        pyconrad.close_all_windows()
