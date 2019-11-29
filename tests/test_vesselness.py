# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""


try:
    import pyconrad.autoinit
except Exception:
    import unittest.mock
    pyconrad = unittest.mock.MagicMock()
import numpy as np
import pytest
import sympy

from pystencils_reco.vesselness import eigenvalues_3d, eigenvalues_3d_3x3_algorithm

pytest.importorskip('tensorflow')


def test_vesselness():
    import tensorflow as tf

    image0 = tf.random.uniform((20, 30, 40, 3, 3))
    eigenvalues_tf, _ = tf.linalg.eigh(image0)

    sorted_eigenvalues = tf.sort(eigenvalues_tf, axis=-1)

    l1 = sorted_eigenvalues[..., -1]
    l2 = sorted_eigenvalues[..., -2]
    l3 = sorted_eigenvalues[..., -3]

    import numpy as np

    image = np.random.rand(30, 40, 50).astype(np.float32)
    result = np.random.rand(30, 40, 50, 3).astype(np.float32)

    assignments = eigenvalues_3d(result, image, image, image, image, image, image)
    print(assignments)
    kernel = assignments.compile(use_auto_for_assignments=True)
    kernel(eigenvaluefield=result, xx=image, xy=image, yy=image, xz=image, zz=image, yz=image)

    import pycuda.autoinit
    from pycuda.gpuarray import to_gpu

    image = to_gpu(np.random.rand(30, 40, 50).astype(np.float32))
    result = to_gpu(np.random.rand(30, 40, 50, 3).astype(np.float32))

    assignments = eigenvalues_3d(result, image, image, image, image, image, image)
    print(assignments)
    kernel = assignments.compile(use_auto_for_assignments=True)
    kernel(eigenvaluefield=result, xx=image, xy=image, yy=image, xz=image, zz=image, yz=image)

    image = tf.random.normal((30, 40, 50))
    result = tf.random.normal((30, 40, 50, 3))

    assignments = eigenvalues_3d(result, image, image, image, image, image, image)
    print(assignments)
    kernel = assignments.compile(use_auto_for_assignments=True, target='cpu')
    eigenvalues = kernel(xx=image, xy=image, yy=image, xz=image, zz=image, yz=image)
    print(eigenvalues.shape)

    import pickle
    pickle.loads(pickle.dumps(kernel))


@pytest.mark.parametrize('target', ('cpu',))
def test_grad(target):
    import tensorflow as tf

    image = tf.random.normal((30, 40, 50))
    result = tf.random.normal((30, 40, 50, 3))

    assignments = eigenvalues_3d(result, image, image, image, image, image, image)
    print(assignments)
    kernel = assignments.compile(use_auto_for_assignments=True, target=target)
    eigenvalues = kernel(xx=image, xy=image, yy=image, xz=image, zz=image, yz=image)
    print(eigenvalues.shape)

    pyconrad.imshow(eigenvalues)


@pytest.mark.parametrize('target', ('cpu',))
def test_3x3(target):
    import tensorflow as tf

    xx = tf.random.normal((30, 40, 50))
    xy = tf.random.normal((30, 40, 50))
    xz = tf.random.normal((30, 40, 50))
    yy = tf.random.normal((30, 40, 50))
    yz = tf.random.normal((30, 40, 50))
    zz = tf.random.normal((30, 40, 50))
    eig1 = tf.random.normal((30, 40, 50))
    eig2 = tf.random.normal((30, 40, 50))
    eig3 = tf.random.normal((30, 40, 50))

    assignments = eigenvalues_3d_3x3_algorithm(eig1, eig2, eig3, xx, xy, xz, yy, yz, zz)
    print(assignments)
    kernel = assignments.compile(use_auto_for_assignments=True, target=target)
    eig1, eig2, eig3 = kernel(xx=xx, xy=xy, yy=yy, xz=xz, zz=zz, yz=yz)


@pytest.mark.parametrize('target', ('cpu',))
def test_3x3_gradient_check(target):
    import tensorflow as tf

    shape = (3, 4, 5)
    xx = tf.random.normal(shape)
    xy = tf.random.normal(shape)
    xz = tf.random.normal(shape)
    yy = tf.random.normal(shape)
    yz = tf.random.normal(shape)
    zz = tf.random.normal(shape)
    eig1 = tf.random.normal(shape)
    eig2 = tf.random.normal(shape)
    eig3 = tf.random.normal(shape)

    assignments = eigenvalues_3d_3x3_algorithm(eig1, eig2, eig3, xx, xy, xz, yy, yz, zz)
    print(assignments)
    kernel = assignments.compile(target=target)
    def fun(xx, xy, xz, yy, yz, zz):

        eig1, eig2, eig3 = kernel(xx=xx, xy=xy, yy=yy, xz=xz, zz=zz, yz=yz)
        return tf.stack([eig1, eig2, eig3])

    theoretical, numerical = tf.test.compute_gradient(
        fun,
        [xx, xy, xz, yy, yz, zz],
        delta=0.001
    )
    # assert np.allclose(theoretical[0], numerical[0])
    print(theoretical)
    print(numerical)
    pyconrad.imshow(theoretical)
    pyconrad.imshow(numerical)


@pytest.mark.parametrize('target', ('cpu',))
def test_3x3_lambdify(target):
    import tensorflow as tf

    shape = (3, 4, 5)
    xx = tf.random.normal(shape)
    xy = tf.random.normal(shape)
    xz = tf.random.normal(shape)
    yy = tf.random.normal(shape)
    yz = tf.random.normal(shape)
    zz = tf.random.normal(shape)
    eig1 = tf.random.normal(shape)
    eig2 = tf.random.normal(shape)
    eig3 = tf.random.normal(shape)

    assignments = eigenvalues_3d_3x3_algorithm(eig1, eig2, eig3, xx, xy, xz, yy, yz, zz)
    print(assignments)
    symbols = sympy.symbols('xx xy xz yy yz zz')
    kernel = assignments.lambdify(symbols, module='tensorflow')
    eig1, eig2, eig3 = kernel(xx, xy, xz, yy, yz, zz)

    def fun(xx, xy, xz, yy, yz, zz):

        eig1, eig2, eig3 = kernel(xx=xx, xy=xy, yy=yy, xz=xz, zz=zz, yz=yz)
        return tf.stack([eig1, eig2, eig3])

    theoretical, numerical = tf.test.compute_gradient(
        fun,
        [xx, xy, xz, yy, yz, zz],
        delta=0.001
    )

    print(theoretical)
    print(numerical)
    pyconrad.imshow(theoretical)
    pyconrad.imshow(numerical)
