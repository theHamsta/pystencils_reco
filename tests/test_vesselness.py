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

from pystencils_autodiff.field_tensor_conversion import create_field_from_array_like
from pystencils_reco.vesselness import eigenvalues_3d, eigenvalues_3x3

pytest.importorskip('tensorflow')


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

    assignments = eigenvalues_3x3(eig1, eig2, eig3, xx, xy, xz, yy, yz, zz)
    print(assignments)
    kernel = assignments.compile(use_auto_for_assignments=True, target=target)
    eig1, eig2, eig3 = kernel(xx=xx, xy=xy, yy=yy, xz=xz, zz=zz, yz=yz)
    assignments = eigenvalues_3d(eig1, eig2, eig3, xx, xy, xz, yy, yz, zz)
    print(assignments)
    kernel = assignments.compile(use_auto_for_assignments=True, target=target)
    eig1, eig2, eig3 = kernel(xx=xx, xy=xy, yy=yy, xz=xz, zz=zz, yz=yz)


@pytest.mark.parametrize('target', ('cpu',))
def test_3x3_gradient_check(target):
    import tensorflow as tf

    shape = (3, 4, 5)
    # xx = tf.random.normal(shape, dtype=tf.float64)
    # xy = tf.random.normal(shape, dtype=tf.float64)
    # xz = tf.random.normal(shape, dtype=tf.float64)
    # yy = tf.random.normal(shape, dtype=tf.float64)
    # yz = tf.random.normal(shape, dtype=tf.float64)
    # zz = tf.random.normal(shape, dtype=tf.float64)
    # eig1 = tf.random.normal(shape, dtype=tf.float64)
    # eig2 = tf.random.normal(shape, dtype=tf.float64)
    # eig3 = tf.random.normal(shape, dtype=tf.float64)

    symmetric = tf.random.uniform((*shape, 3, 3), dtype=tf.float64)
    image0 = 1 + (tf.ones((*shape, 3, 3), dtype=tf.float64) + 0.2 *
                  (tf.transpose(symmetric, perm=(0, 1, 2, 4, 3)) + symmetric))
    xx = image0[..., 0, 0]
    yy = image0[..., 1, 1]
    zz = image0[..., 2, 2]
    xy = image0[..., 0, 1]
    xz = image0[..., 0, 2]
    yz = image0[..., 1, 2]
    # xx = tf.ones(shape, dtype=tf.float64)
    # xy = tf.ones(shape, dtype=tf.float64)
    # xz = tf.ones(shape, dtype=tf.float64)
    # yy = tf.ones(shape, dtype=tf.float64)
    # yz = tf.ones(shape, dtype=tf.float64)
    # zz = tf.ones(shape, dtype=tf.float64)
    eig1 = tf.ones(shape, dtype=tf.float64)
    eig2 = tf.ones(shape, dtype=tf.float64)
    eig3 = tf.ones(shape, dtype=tf.float64)

    assignments = eigenvalues_3x3(eig1, eig2, eig3, xx, xy, xz, yy, yz, zz)
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
    # print(theoretical)
    # print(numerical)
    # pyconrad.imshow(theoretical)
    # pyconrad.imshow(numerical)
    # pyconrad.imshow(numerical[0]-theoretical[0])
    assert np.allclose(theoretical[0], numerical[0], rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize('repetition', range(1))
@pytest.mark.parametrize('target', ('gpu',))
def test_3x3_gradient_check_torch(target, repetition):
    import torch
    torch.set_default_dtype(torch.double)

    shape = (20, 40, 50)
    xx = torch.randn(*shape, requires_grad=True)
    xy = torch.randn(*shape, requires_grad=True)
    xz = torch.randn(*shape, requires_grad=True)
    yy = torch.randn(*shape, requires_grad=True)
    yz = torch.randn(*shape, requires_grad=True)
    zz = torch.randn(*shape, requires_grad=True)
    eig1 = torch.randn(*shape)
    eig2 = torch.randn(*shape)
    eig3 = torch.randn(*shape)

    if target == 'gpu':
        xx = xx.cuda()
        xy = xy.cuda()
        xz = xz.cuda()
        yy = yy.cuda()
        yz = yz.cuda()
        zz = zz.cuda()
        eig1 = eig1.cuda()
        eig2 = eig2.cuda()
        eig3 = eig3.cuda()

    assignments = eigenvalues_3x3(eig1, eig2, eig3, xx, xy, xz, yy, yz, zz)
    print(assignments)
    kernel = assignments.compile(target=target)

    assert torch.autograd.gradcheck(kernel.apply, [xx, xy, xz, yy, yz, zz])


@pytest.mark.parametrize('target', ('cpu',))
@pytest.mark.xfail(reason="Bug in sympy lambdify with ceil", strict=False)
def test_3x3_lambdify(target):
    import tensorflow as tf

    shape = (3, 4, 5)
    xx = tf.random.normal(shape, dtype=tf.float64)
    xy = tf.random.normal(shape, dtype=tf.float64)
    xz = tf.random.normal(shape, dtype=tf.float64)
    yy = tf.random.normal(shape, dtype=tf.float64)
    yz = tf.random.normal(shape, dtype=tf.float64)
    zz = tf.random.normal(shape, dtype=tf.float64)
    eig1 = tf.random.normal(shape, dtype=tf.float64)
    eig2 = tf.random.normal(shape, dtype=tf.float64)
    eig3 = tf.random.normal(shape, dtype=tf.float64)

    assignments = eigenvalues_3x3(eig1, eig2, eig3, xx, xy, xz, yy, yz, zz)
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


@pytest.mark.parametrize('target', ('cpu',))
def test_check_forward(target):
    import tensorflow as tf

    shape = (5, 7, 6)

    # image0 = 1 + tf.random.uniform((*shape, 3, 3))
    symmetric = tf.random.uniform((*shape, 3, 3))
    image0 = 1 + tf.ones((*shape, 3, 3)) + 0.2 * (tf.transpose(symmetric, perm=(0, 1, 2, 4, 3)) + symmetric)
    xx = image0[..., 0, 0]
    yy = image0[..., 1, 1]
    zz = image0[..., 2, 2]
    xy = image0[..., 0, 1]
    xz = image0[..., 0, 2]
    yz = image0[..., 1, 2]

    eigenvalues_tf, vectors = tf.linalg.eigh(image0)
    e1_field = create_field_from_array_like('e1', xx)
    e2_field = create_field_from_array_like('e2', xx)
    e3_field = create_field_from_array_like('e3', xx)

    assignments = eigenvalues_3x3(e1_field, e2_field, e3_field, xx, xy, xz, yy, yz, zz)
    # assignments = eigenvalues_3x3(e1_field, e2_field, e3_field, xx, xy, xz, yy, yz, zz)
    kernel = assignments.compile(target=target)

    r1, r2, r3 = kernel(xx=xx, yy=yy, zz=zz, xy=xy, xz=xz, yz=yz)

    sorted_own = np.array(tf.sort(tf.stack([r1, r2, r3]), axis=0))
    sorted_tf = np.sort(np.moveaxis(np.array(eigenvalues_tf), -1, 0), axis=0)
    print(sorted_own.shape)
    print(sorted_tf.shape)

    print(sorted_own)
    print(sorted_tf)

    # import pyconrad.autoinit
    # pyconrad.imshow(sorted_own)
    # pyconrad.imshow(sorted_tf)
    # pyconrad.imshow(sorted_own - sorted_tf)

    max_diff = np.max(np.abs(sorted_own - sorted_tf))
    print(max_diff)

    assert np.allclose(sorted_own, sorted_tf, rtol=1e-2, atol=1e-3)


def test_check_forward_pycuda():
    pytest.importorskip('pycuda')
    import pycuda.autoinit
    from pycuda.gpuarray import to_gpu

    shape = (200, 200, 200)

    xx = to_gpu(np.ones(shape))
    yy = to_gpu(np.ones(shape))
    zz = to_gpu(np.ones(shape))
    xy = to_gpu(np.ones(shape))
    xz = to_gpu(np.ones(shape))
    yz = to_gpu(np.ones(shape))

    e1_field = to_gpu(np.ones(shape))
    e2_field = to_gpu(np.ones(shape))
    e3_field = to_gpu(np.ones(shape))

    assignments = eigenvalues_3d(e1_field, e2_field, e3_field, xx, xy, xz, yy, yz, zz)
    # assignments = eigenvalues_3x3(e1_field, e2_field, e3_field, xx, xy, xz, yy, yz, zz)
    kernel = assignments.compile()

    kernel()


test_3x3('cpu')
