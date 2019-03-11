# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from pystencils_reco.stencils import BallStencil, BoxStencil


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


def test_BoxStencil_non_quadratic():
    stencil = BoxStencil((3, 5))
    print(stencil)


def main():
    test_BoxStencil()
    test_BallStencil()
    test_BoxStencil_non_quadratic()


if __name__ == '__main__':
    main()
