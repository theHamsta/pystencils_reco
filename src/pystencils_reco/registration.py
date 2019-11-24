# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import sympy as sp

import pystencils
from pystencils.rng import PhiloxFourFloats
from pystencils_autodiff.transformations import get_random_sampling
from pystencils_reco import crazy, fixed_boundary_handling


@fixed_boundary_handling
@crazy
def autocorrelation(x: {'field_type': pystencils.FieldType.CUSTOM},
                    y,
                    r_xx,
                    r_yy,
                    r_xy,
                    stencil):
    print(locals())

    sum_xx = []
    sum_yy = []
    sum_xy = []

    for s in stencil:
        sum_xx.append(x[s] * x.center)
        sum_yy.append(y[s] * y.center)
        sum_xy.append(y[s] * x.center)

    cross = sp.Symbol('cross')
    return {
        cross: sp.Add(*sum_xy),
        r_xy.center: cross,
        r_xx.center: sp.Piecewise((sp.Add(*sum_xx), sp.Abs(cross) > 1e-3), (1., True)),
        r_yy.center: sp.Piecewise((sp.Add(*sum_yy), sp.Abs(cross) > 1e-3), (1., True)),
    }


@crazy
def autocorrelation_random_sampling(x,
                                    y,
                                    auto_correlation,
                                    time_step=pystencils.data_types.TypedSymbol(
                                        'time_step', pystencils.data_types.create_type('int32')),
                                    eps=1e-3):

    assert auto_correlation.spatial_dimensions == 1

    random_floats = PhiloxFourFloats(1, time_step)
    random_point = get_random_sampling(random_floats.result_symbols[:x.spatial_dimensions], y.aabb_min, y.aabb_max)

    x_point = x.interpolated_access(x.physical_to_index(random_point))
    y_point = y.interpolated_access(y.physical_to_index(random_point))

    xx = x_point * x_point
    yy = y_point * y_point
    xy = y_point * x_point

    return [
        random_floats,
        pystencils.Assignment(auto_correlation.center, xy ** 2 / (xx * yy + eps))
    ]
