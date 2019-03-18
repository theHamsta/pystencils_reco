# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""
Implements a generic forward and backprojection projections
"""

import sympy

import pystencils
import pystencils.astnodes
import pystencils_reco._geometry


def forward_projection(input_volume_field, output_projections_field, projection_matrix, step_size=1):

    volume_texture = pystencils.astnodes.TextureCachedField(input_volume_field)
    ndim = input_volume_field.spatial_dimensions
    projection_matrix = pystencils_reco.ProjectiveMatrix(projection_matrix)

    texture_coordinates = sympy.Matrix(pystencils.typed_symbols(f'_t:{ndim}', 'float32'))
    u = output_projections_field.physical_coordinates_staggered
    x = input_volume_field.create_staggered_physical_coordinates(texture_coordinates)

    eqn = projection_matrix @ x - u
    ray_equations = sympy.solve(eqn, texture_coordinates, rational=False)
    ray_equations = sympy.Matrix([ray_equations[t] for t in texture_coordinates[:-1]] + [texture_coordinates[-1]])

    t = texture_coordinates[-1]

    projection_vector = sympy.diff(ray_equations, t)
    projection_vector_norm = projection_vector.norm()
    projection_vector /= projection_vector_norm

    conditions = pystencils_reco._geometry.coordinate_in_field_conditions(
        input_volume_field, texture_coordinates)

    central_ray = sympy.Matrix(projection_matrix.nullspace()[0][:input_volume_field.spatial_dimensions])
    central_ray /= central_ray.norm()

    intersection_candidates = []
    for i in range(ndim):
        solution_min = sympy.solve(ray_equations[i], t, rational=False)
        solution_max = sympy.solve(ray_equations[i] - input_volume_field.spatial_shape[i],
                                   t,
                                   rational=False)
        intersection_candidates.extend(solution_min + solution_max)

    intersection_point1 = sympy.Piecewise(
        *[(f, sympy.And(*conditions).subs({t: f})) for f in intersection_candidates], (0.5, True))
    intersection_point2 = sympy.Piecewise(*[(f, sympy.And(*conditions).subs({t: f}))
                                            for f in reversed(intersection_candidates)], (0.5, True))

    min_t = sympy.Piecewise((intersection_point1, intersection_point1 < intersection_point2),
                            (intersection_point2, True))
    max_t = sympy.Piecewise((intersection_point1, intersection_point1 > intersection_point2),
                            (intersection_point2, True))
    # num_steps = sympy.ceiling(max_t-min_t) / step_size

    line_integral, num_steps, min_t_tmp, max_t_tmp, intensity_weighting = pystencils.data_types.typed_symbols(
        'line_integral, num_steps, min_t_tmp, max_t_tmp, intensity_weighting', 'float32')
    i = pystencils.data_types.TypedSymbol('i', 'int32')

    assignments = pystencils_reco.AssignmentCollection({
        min_t_tmp: min_t,
        max_t_tmp: max_t,
        num_steps: sympy.ceiling(max_t_tmp - min_t_tmp / step_size),
        line_integral: sympy.Sum(volume_texture.at(ray_equations.subs({t: min_t_tmp + i * step_size}).simplify()),
                                 (i, 0, num_steps)),
        intensity_weighting: projection_vector.dot(central_ray) ** 2,
        output_projections_field.center(): (line_integral * step_size * intensity_weighting)
    })

    return assignments


def backward_projection(input_projection, output_volume, projection_matrix, normalization):
    projection_matrix = pystencils_reco.ProjectiveMatrix(projection_matrix)
    assignments = pystencils_reco.resampling.generic_spatial_matrix_transform(
        input_projection, output_volume, None, inverse_matrix=projection_matrix)
    normalized_assignments = [a / normalization for a in assignments.all_assignments]

    return pystencils_reco.AssignmentCollection(normalized_assignments)
