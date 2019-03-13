# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pystencils
from pystencils_reco.resampling import (generic_spatial_matrix_transform,
                                        rotation_transform, scale_transform)


def test_scaling():

    for ndim in range(1, 5):
        for scale in (0.5, [(s+1)*0.1 for s in range(ndim)]):
            x, y = pystencils.fields('x,y: float32[%id]' % ndim)
            transform = scale_transform(x, y, scale)
            print(transform)


def test_rotation():

    for ndim in (2, 3):
        for angle in (0.4, -0.8):
            for axis in range(3):
                x, y = pystencils.fields('x,y: float32[%id]' % ndim)
                transform = rotation_transform(x, y, angle, axis)
                print(transform)


def main():
    test_scaling()
    test_rotation()


if __name__ == '__main__':
    main()
