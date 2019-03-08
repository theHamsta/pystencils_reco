# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""
Implements morphological operations
"""
import sympy

import pystencils


def binary_erosion(input_field, output_field, stencil):
    return _morphological(input_field, output_field, stencil, is_erosion=True)


def binary_dilation(input_field, output_field, stencil):
    return _morphological(input_field, output_field, stencil, is_erosion=False)


def _morphological(input_field, output_field, stencil, is_erosion):
    sum = 0
    for s in stencil:
        if is_erosion:
            pixel_ok = sympy.Piecewise((1, input_field[s] > 0), (0, True))
        else:
            pixel_ok = sympy.Piecewise((1, input_field[s] <= 0), (0, True))

        sum += pixel_ok

    if is_erosion:
        central_pixel = sympy.Piecewise((1, sum >= len(stencil)),  (0, True))
    else:
        central_pixel = sympy.Piecewise((0, sum >= len(stencil)),  (1, True))

    assignments = pystencils.AssignmentCollection({
        output_field.center(): central_pixel
    }, {})

    return assignments
