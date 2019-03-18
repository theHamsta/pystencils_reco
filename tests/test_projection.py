# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import os

import numpy as np
import sympy

import pystencils
from pystencils_reco.projection import forward_projection


def test_projection():
    volume = pystencils.fields('volume: float32[100,200,300]')
    projections = pystencils.fields('projections: float32[2D]')

    projection_matrix = sympy.Matrix([[-289.0098977737411, -1205.2274801832275, 0.0, 186000.0],
                                      [-239.9634468375339, - 4.188577544948043, 1200.0, 144000.0],
                                      [-0.9998476951563913, -0.01745240643728351, 0.0, 600.0]])

    kernel = forward_projection(volume, projections, projection_matrix).compile(target='gpu')
    print(kernel.code)

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
    kernel = forward_projection(volume, projections, projection_matrix).compile(target='gpu')
    print(kernel.code)


def project_shepp_logan():
    import pyconrad.autoinit
    import pycuda.autoinit
    from pycuda.gpuarray import to_gpu, GPUArray

    from edu.stanford.rsl.conrad.phantom import NumericalSheppLogan3D
    phantom3d = NumericalSheppLogan3D(100, 100, 100).getNumericalSheppLoganPhantom()
    pyconrad.imshow(phantom3d, 'phantom')

    m0 = sympy.Matrix([[1, 0, 0],
                       [0, 0, 1]])
    m1 = sympy.Matrix([[-289.0098977737411, -1205.2274801832275, 0.0, 186000.0],
                       [-239.9634468375339, - 4.188577544948043, 1200.0, 144000.0],
                       [-0.9998476951563913, -0.01745240643728351, 0.0, 600.0]])
    m2 = sympy.Matrix([[1, 0, 0],
                       [0, 1, 0]])
    m3 = sympy.Matrix([[0, 1, 0],
                       [0, 1, 1]])

    for i, projection_matrix in enumerate((m0, m1, m2, m3)):

        volume = pystencils.fields('volume: float32[100,100,100]')
        projections = pystencils.fields('projections: float32[1024,960]')
        projections_stack = pystencils.fields('projections(200): float32[2D]')
        volume.set_coordinate_origin_to_field_center()
        volume.coordinate_transform = sympy.rot_axis2(0.2)
        # volume.coordinate_transform = sympy.rot_axis3(0.1)
        volume.coordinate_transform = 3 * volume.coordinate_transform
        projections.set_coordinate_origin_to_field_center()

        kernel = forward_projection(volume, projections, projection_matrix, step_size=1)
        print(kernel)
        kernel = kernel.compile('gpu')
        print(kernel.code)

        volume_gpu = to_gpu(np.ascontiguousarray(phantom3d.as_numpy(), np.float32))
        projection_gpu = GPUArray(projections.spatial_shape, np.float32)

        kernel(volume=volume_gpu, projections=projection_gpu)

        pyconrad.imshow(volume_gpu, 'volume')
        pyconrad.imshow(projection_gpu, 'projections' + str(i))

    for i, projection_matrix in enumerate((m0, m1, m2, m3)):
        angle = pystencils.typed_symbols('angle', 'float32')

        volume = pystencils.fields('volume: float32[100,100,100]')
        projections = pystencils.fields('projections: float32[1024,960]')
        projections_stack = pystencils.fields('projections(200): float32[2D]')
        volume.set_coordinate_origin_to_field_center()
        volume.coordinate_transform = sympy.rot_axis2(angle)
        # volume.coordinate_transform = sympy.rot_axis3(0.1)
        volume.coordinate_transform = 3 * volume.coordinate_transform
        projections.set_coordinate_origin_to_field_center()

        kernel = forward_projection(volume, projections, projection_matrix, step_size=1)
        print(kernel)
        kernel = kernel.compile('gpu')
        print(kernel.code)

        volume_gpu = to_gpu(np.ascontiguousarray(phantom3d.as_numpy(), np.float32))
        projection_gpu = GPUArray(projections.spatial_shape, np.float32)

        for phi in np.arange(0, np.pi, np.pi / 100):
            kernel(volume=volume_gpu, projections=projection_gpu, angle=phi)
            pyconrad.imshow(projection_gpu, 'rotation!')
        pyconrad.close_all_windows()

def project_shepp_logan():
    import pyconrad.autoinit
    import pycuda.autoinit
    from pycuda.gpuarray import to_gpu, GPUArray

    from edu.stanford.rsl.conrad.phantom import NumericalSheppLogan3D
    phantom3d = NumericalSheppLogan3D(100, 100, 100).getNumericalSheppLoganPhantom()
    pyconrad.imshow(phantom3d, 'phantom')

    projection_matrices = pyconrad.config.get_conf.ge
    for i, projection_matrix in enumerate((m0, m1, m2, m3)):

        volume = pystencils.fields('volume: float32[100,100,100]')
        projections = pystencils.fields('projections: float32[1024,960]')
        projections_stack = pystencils.fields('projections(200): float32[2D]')
        volume.set_coordinate_origin_to_field_center()
        volume.coordinate_transform = sympy.rot_axis2(0.2)
        # volume.coordinate_transform = sympy.rot_axis3(0.1)
        volume.coordinate_transform = 3 * volume.coordinate_transform
        projections.set_coordinate_origin_to_field_center()

        kernel = forward_projection(volume, projections, projection_matrix, step_size=1)
        print(kernel)
        kernel = kernel.compile('gpu')
        print(kernel.code)

        volume_gpu = to_gpu(np.ascontiguousarray(phantom3d.as_numpy(), np.float32))
        projection_gpu = GPUArray(projections.spatial_shape, np.float32)

        kernel(volume=volume_gpu, projections=projection_gpu)

        pyconrad.imshow(volume_gpu, 'volume')
        pyconrad.imshow(projection_gpu, 'projections' + str(i))

    for i, projection_matrix in enumerate((m0, m1, m2, m3)):
        angle = pystencils.typed_symbols('angle', 'float32')

        volume = pystencils.fields('volume: float32[100,100,100]')
        projections = pystencils.fields('projections: float32[1024,960]')
        projections_stack = pystencils.fields('projections(200): float32[2D]')
        volume.set_coordinate_origin_to_field_center()
        volume.coordinate_transform = sympy.rot_axis2(angle)
        # volume.coordinate_transform = sympy.rot_axis3(0.1)
        volume.coordinate_transform = 3 * volume.coordinate_transform
        projections.set_coordinate_origin_to_field_center()

        kernel = forward_projection(volume, projections, projection_matrix, step_size=1)
        print(kernel)
        kernel = kernel.compile('gpu')
        print(kernel.code)

        volume_gpu = to_gpu(np.ascontiguousarray(phantom3d.as_numpy(), np.float32))
        projection_gpu = GPUArray(projections.spatial_shape, np.float32)

        for phi in np.arange(0, np.pi, np.pi / 100):
            kernel(volume=volume_gpu, projections=projection_gpu, angle=phi)
            pyconrad.imshow(projection_gpu, 'rotation!')
        pyconrad.close_all_windows()

def main():
    test_projection()
    project_shepp_logan()


if __name__ == '__main__':
    main()
