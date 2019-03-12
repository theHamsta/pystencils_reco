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
# from diofant.geometry import Line, Point, Po
from diofant.solvers.inequalities import reduce_inequalities

import pystencils.astnodes
import pystencils_reco._geometry


def forward_projection(input_volume_field, output_projections_field, projection_matrix, step_size=1):

    volume_texture = pystencils.astnodes.TextureCachedField(input_volume_field)
    ndim = input_volume_field.spatial_dimensions

    x, y, z = sympy.symbols('x,y,z')
    u, v, t = sympy.symbols('u,v,t')
    lhs = projection_matrix * sympy.Matrix([x, y, z, 1])
    rhs = [u*t, v*t, t]

    ray_equations = sympy.solve([a-b for a, b in zip(lhs, rhs)], [x, y, z], rational=False)
    conditions = pystencils_reco._geometry.coordinate_in_field_conditions(
        input_volume_field, [ray_equations[x], ray_equations[y], ray_equations[z]])

    intersection_candidates = []
    for i, s in enumerate((x, y, z)):
        intersection_candidates.append(sympy.solve(ray_equations[s], [t], rational=False)[0])
        intersection_candidates.append(sympy.solve(
            ray_equations[s]-input_volume_field.spatial_shape[i], [t], rational=False)[0])

    itersection_point1 = sympy.Piecewise(
        *[(f, sympy.And(*conditions).subs({t: f})) for f in intersection_candidates], (0, True))
    itersection_point2 = sympy.Piecewise(*[(f, sympy.And(*conditions).subs({t: f}))
                                           for f in reversed(intersection_candidates)], (0, True))

    min_t = sympy.Piecewise((itersection_point1, itersection_point1 < itersection_point2), (itersection_point2, True))
    max_t = sympy.Piecewise((itersection_point1, itersection_point1 > itersection_point2), (itersection_point2, True))
    # num_steps = sympy.ceiling(max_t-min_t) / step_size

    line_integral, i, num_steps, min_t_tmp, max_t_tmp,  = sympy.symbols(
        'line_integral,i, num_steps, min_t_tmp, max_t_tmp')

    texture_access_index = sympy.symbols('texture_access_index:3')
    # point1 = sympy.geometry.Point([ray_equations[s].subs({t: 0}) for s in (x, y, z)])
    # point2 = sympy.geometry.Point([ray_equations[s].subs({t: 100}) for s in (x, y, z)])
    # projection_ray = sympy.geometry.Line(point1, point2)
    # volume_box = pystencils_reco._geometry.get_field_box_boundary(input_volume_field)
    # volume_box = pystencils_reco._geometry.get_field_box_corner_points(input_volume_field)
    # ray_intersection_points = sympy.geometry.intersection(volume_box, projection_ray)

    # box = pystencils_reco._geometry.get_field_box(input_volume_field)
    # print(ray_intersection_points)

    # print(in_box_conditions)
    # ray_box_intersections = sympy.solve(ray_equations.values() + in_box_conditions)
    # print(ray_box_intersections)

    # point1 = sympy.geometry.Point([ray_equations[s].subs({t: 0}) for s in (x, y, z)])
    # point2 = sympy.geometry.Point([ray_equations[s].subs({t: 100}) for s in (x, y, z)])
    # projection_ray = sympy.geometry.Line(point1, point2)
    # volume_box = pystencils_reco._geometry.get_field_box(input_volume_field)
    # ray_segment = volume_box.interection(projection_ray)
    # print(ray_segment)

    u = pystencils.x_staggered
    v = pystencils.y_staggered

    assignments = {
        output_projections_field.center(): line_integral * step_size
    }

    sub_expressions = {
        texture_access_index[index]: ray_equations[s].subs({t: min_t_tmp + i * step_size}) for index, s in enumerate((x, y, z))
    }
    sub_expressions.update({
        min_t_tmp: min_t,
        max_t_tmp: max_t,
        num_steps: sympy.ceiling(max_t_tmp-min_t_tmp / step_size),
        line_integral: sympy.Sum(volume_texture.at([texture_access_index[i] for i in range(ndim)]), (i, 0, num_steps))
        })
    return pystencils_reco.AssignmentCollection(assignments, sub_expressions)
