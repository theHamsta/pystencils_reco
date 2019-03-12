# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

from pystencils_reco.projection import forward_projection
import sympy
import pystencils


def test_projection():
    volume = pystencils.fields('volume: float32[3D]')
    projections = pystencils.fields('projections: float32[2D]')

    projection_matrix = sympy.Matrix([[-289.0098977737411, -1205.2274801832275, 0.0, 186000.0],
                                      [-239.9634468375339, - 4.188577544948043, 1200.0, 144000.0],
                                      [-0.9998476951563913, -0.01745240643728351, 0.0, 600.0]])

    forward_projection(volume, projections, projection_matrix)


def main():
    test_projection()


if __name__ == '__main__':
    main()
