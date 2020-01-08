# -*- coding: utf-8 -*-
#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import sympy as sp

import pystencils


def extend_to_size_of_other_field(this_field: pystencils.Field, other_field: pystencils.Field):
    this_field.coordinate_transform = sp.DiagMatrix(sp.Matrix([this_field.spatial_shape[i]
                                                               / other_field.spatial_shape[i]
                                                               for i in range(len(this_field.spatial_shape))]))
