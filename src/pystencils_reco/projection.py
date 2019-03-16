# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""
Implements a generic forward and backprojection projections
"""

import sympy

import pystencils.astnodes
import pystencils_reco._geometry


def forward_projection(input_volume_field, output_projections_field, projection_matrix, step_size=1):

    volume_texture = pystencils.astnodes.TextureCachedField(input_volume_field)

    x, y, z = sympy.symbols('x,y,z')
    u, v, t = sympy.symbols('u,v,t')
    lhs = sympy.Matrix(projection_matrix) * sympy.Matrix([x, y, z, 1])
    rhs = [u*t, v*t, t]

    ray_equations = sympy.solve([a-b for a, b in zip(lhs, rhs)], [x, y, z], rational=False)
    projection_vector = sympy.Matrix([sympy.diff(ray_equations[s], t) for s in (x, y, z)])
    projection_vector_norm = projection_vector.norm()
    projection_vector /= projection_vector_norm
    conditions = pystencils_reco._geometry.coordinate_in_field_conditions(
        input_volume_field, [ray_equations[x], ray_equations[y], ray_equations[z]])
    # source_position = projection_matrix.nullspace()
    # assert source_position[3] == 0

    intersection_candidates = []
    for i, s in enumerate((x, y, z)):
        intersection_candidates.append(sympy.solve(ray_equations[s], [t], rational=False)[0])
        intersection_candidates.append(sympy.solve(
            ray_equations[s]-input_volume_field.spatial_shape[i], [t], rational=False)[0])

    intersection_point1 = sympy.Piecewise(
        *[(f, sympy.And(*conditions).subs({t: f})) for f in intersection_candidates], (0, True))
    intersection_point2 = sympy.Piecewise(*[(f, sympy.And(*conditions).subs({t: f}))
                                            for f in reversed(intersection_candidates)], (0, True))

    min_t = sympy.Piecewise((intersection_point1, intersection_point1 < intersection_point2),
                            (intersection_point2, True))
    max_t = sympy.Piecewise((intersection_point1, intersection_point1 > intersection_point2),
                            (intersection_point2, True))
    # num_steps = sympy.ceiling(max_t-min_t) / step_size

    line_integral, i, num_steps, min_t_tmp, max_t_tmp, intensity_weighting = sympy.symbols(
        'line_integral,i, num_steps, min_t_tmp, max_t_tmp, intensity_weighting')
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

    # step_size *= projection_vector_norm

    assignments = pystencils_reco.AssignmentCollection({
        min_t_tmp: min_t.subs({u: pystencils.x_staggered, v: pystencils.y_staggered}),
        max_t_tmp: max_t.subs({u: pystencils.x_staggered, v: pystencils.y_staggered}),
        num_steps: sympy.ceiling(max_t_tmp-min_t_tmp / step_size),
        line_integral: sympy.Sum(volume_texture.at(
            [ray_equations[s].subs({t: min_t_tmp + i * step_size,
                                    u: pystencils.x_staggered,
                                    v: pystencils.y_staggered}) for s in (x, y, z)]), (i, 0, num_steps)),
        # intensity_weighting: sympy.dot(projection_vector#,
        output_projections_field.center(): (line_integral * step_size)
    })

    return assignments


def backward_projection(input_projection, output_volume, projection_matrix, normalization):
    projection_matrix = pystencils_reco.ProjectiveMatrix(projection_matrix)
    assignments = pystencils_reco.resampling.generic_spatial_matrix_transform(
        input_projection, output_volume, None, inverse_matrix=projection_matrix)
    normalized_assignments = [a / normalization for a in assignments.all_assignments]

    return pystencils_reco.AssignmentCollection(normalized_assignments)
