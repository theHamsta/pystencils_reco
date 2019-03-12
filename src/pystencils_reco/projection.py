# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""
Implements a generic forward and backprojection projections
"""

import diofant.geometry
import sympy
from diofant.solvers.inequalities import reduce_inequalities

import pystencils.astnodes
import pystencils_reco._geometry


def forward_projection(input_volume_field, output_projections_field, projection_matrix, step_size=1):

    volume_texture = pystencils.astnodes.TextureCachedField(input_volume_field)

    x, y, z = sympy.symbols('x,y,z')
    u, v, t = sympy.symbols('u,v,t')
    lhs = projection_matrix * sympy.Matrix([x, y, z, 1])
    rhs = [u*t, v*t, t]

    ray_equations = sympy.solve([a-b for a, b in zip(lhs, rhs)], [x, y, z])
    inequalities = pystencils_reco._geometry.coordinate_in_field_conditions(input_volume_field, [x, y, z])
    # foo = reduce_inequalities([ray_equations[x]]+inequalities, [t])
    # print(foo)

    point1 = sympy.geometry.Point([ray_equations[s].subs({t: 0}) for s in (x, y, z)])
    point2 = sympy.geometry.Point([ray_equations[s].subs({t: 100}) for s in (x, y, z)])
    projection_ray = diofant.geometry.Line(point1, point2)
    volume_box = pystencils_reco._geometry.get_field_box_boundary(input_volume_field)
    ray_segment = diofant.geometry.intersection(volume_box, projection_ray)

    # box = pystencils_reco._geometry.get_field_box(input_volume_field)
    print(ray_segment)

    # print(in_box_conditions)
    # ray_box_intersections = sympy.solve(ray_equations.values() + in_box_conditions)
    # print(ray_box_intersections)

    # point1 = sympy.geometry.Point([ray_equations[s].subs({t: 0}) for s in (x, y, z)])
    # point2 = sympy.geometry.Point([ray_equations[s].subs({t: 100}) for s in (x, y, z)])
    # projection_ray = sympy.geometry.Line(point1, point2)
    # volume_box = pystencils_reco._geometry.get_field_box(input_volume_field)
    # ray_segment = volume_box.interection(projection_ray)
    # print(ray_segment)

    line_integral = sympy.Symbol('line_integral')

    # assignments = pystencils.AssignmentCollection({
    # line_integral: sympy.Sum(volume.at(t*step_size*projection_vector), (t, 0, 100/step_size)),
    # b.center(): line_integral * step_size
    # }, {})
