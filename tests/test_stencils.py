# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from pystencils_reco.stencils import BallStencil, BoxStencil, LineStencil


def test_LineStencil():
    for with_center in (True, False):
        for ndim in range(1, 4):
            for filter_dimension in range(ndim):
                for kernel_size in range(1, 6):
                    stencil = LineStencil(kernel_size, filter_dimension, ndim, with_center=with_center)
                    print(stencil)


def test_BoxStencil():
    for with_center in (True, False):
        for ndim in range(1, 4):
            for kernel_size in (3, 5, 7):
                stencil = BoxStencil(kernel_size, ndim, with_center=with_center)
                print(stencil)


def test_BallStencil():
    for with_center in (True, False):
        for ndim in range(1, 4):
            for radius in range(1, 5):
                stencil = BallStencil(radius, ndim, with_center=with_center)
                print(stencil)


def test_strided():

    for with_center in (True, False):
        for ndim in range(1, 4):
            strides = list(range(2, 2+ndim))
            for radius in (5,):
                stencil = BallStencil(radius, ndim, with_center=with_center).as_strided(strides)
                print(stencil)

    for with_center in (True, False):
        for ndim in range(1, 4):
            strides = [2] * ndim
            for radius in (5,):
                stencil = BallStencil(radius, ndim, with_center=with_center).as_strided(strides)
                print(stencil)


def test_BoxStencil_non_quadratic():
    stencil = BoxStencil((3, 5))
    print(stencil)
