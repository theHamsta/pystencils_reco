# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""
Implements a generic forward and backprojection projections
"""

import types

import sympy

import pystencils
import pystencils.astnodes
import pystencils.autodiff
import pystencils.interpolation_astnodes
import pystencils_reco._geometry
from pystencils.autodiff import AdjointField
from pystencils.cache import disk_cache_no_fallback
from pystencils_reco import crazy


@crazy
def forward_projection(input_volume_field: pystencils.Field,
                       output_projections_field: pystencils.Field,
                       projection_matrix,
                       step_size=1,
                       cubic_bspline_interpolation=False):
    # is_projection_stack = output_projections_field.spatial_dimensions == input_volume_field.spatial_dimensions

    interpolation_mode = 'cubic_spline' if cubic_bspline_interpolation else 'linear'
    volume_texture = pystencils.interpolation_astnodes.Interpolator(input_volume_field,
                                                                    interpolation_mode)
    ndim = input_volume_field.spatial_dimensions
    projection_matrix = pystencils_reco.ProjectiveMatrix(projection_matrix)

    t = pystencils_reco.typed_symbols('_parametrization', 'float32')
    texture_coordinates = sympy.Matrix(pystencils_reco.typed_symbols(f'_t:{ndim}', 'float32'))
    u = output_projections_field.physical_coordinates_staggered
    x = input_volume_field.index_to_physical(texture_coordinates)

    is_perspective = projection_matrix.matrix.cols == ndim + 1

    if is_perspective:
        eqn = projection_matrix @ sympy.Matrix([*x, 1]) - sympy.Matrix([*(t * u), t])
    else:
        # this also works for perspective/cone beam projection (but may lead to instable parametrization)
        eqn = projection_matrix @ x - u
    ray_equations = sympy.solve(eqn, texture_coordinates, rational=False)
    print('Solved')

    if not is_perspective:
        t = [t for t in texture_coordinates if t not in ray_equations.keys()][0]
        assert len(ray_equations.keys()) == ndim - 1, "projection_matrix does not appear to define a projection"
    ray_equations = sympy.Matrix([ray_equations[s] if s != t else t for s in texture_coordinates])

    projection_vector = sympy.diff(ray_equations, t)
    projection_vector_norm = projection_vector.norm()
    projection_vector /= projection_vector_norm

    conditions = pystencils_reco._geometry.coordinate_in_field_conditions(
        input_volume_field, ray_equations)

    central_ray = sympy.Matrix(
        projection_matrix.nullspace()[0][:input_volume_field.spatial_dimensions])
    central_ray /= central_ray.norm()

    intersection_candidates = []
    for i in range(ndim):
        solution_min = sympy.solve(ray_equations[i], t, rational=False)
        solution_max = sympy.solve(ray_equations[i] - input_volume_field.spatial_shape[i],
                                   t,
                                   rational=False)
        intersection_candidates.extend(solution_min + solution_max)

    intersection_point1 = sympy.Piecewise(
        *[(f, sympy.And(*conditions).subs({t: f})) for f in intersection_candidates], (-0, True))
    intersection_point2 = sympy.Piecewise(*[(f, sympy.And(*conditions).subs({t: f}))
                                            for f in reversed(intersection_candidates)], (-0, True))
    assert intersection_point1 != intersection_point2, \
        "The intersections are unconditionally equal, reconstruction volume is not in detector FOV!"

    # perform a integer set analysis here?
    # space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, set=[str(t) for t in texture_coordinates])
    # ray_set = isl.BasicSet.universe(space)
    # for i, t in enumerate(texture_coordinates):
    #    # dafaq?
    #    ray_set.add_constraint(isl.Constraint.ineq_from_names(space, {str(texture_coordinates): 1}))
    #    ray_set.add_constraint(isl.Constraint.ineq_from_names(space,
    #                                                        # {1: -input_volume_field.shape[i],
    # str(texture_coordinates): -1}))
    #    ray_set.add_constraint(isl.Constraint.eq_from_name(space, ray_equations[i].subs({ #TODO

    min_t = sympy.Min(intersection_point1, intersection_point2)
    max_t = sympy.Max(intersection_point1, intersection_point2)
    # parametrization_dim = list(ray_equations).index(t)
    # min_t = 0
    # max_t = input_volume_field.spatial_shape[parametrization_dim]

    line_integral, num_steps, min_t_tmp, max_t_tmp, intensity_weighting, step = pystencils.data_types.typed_symbols(
        'line_integral, num_steps, min_t_tmp, max_t_tmp, intensity_weighting, step', 'float32')
    i = pystencils.data_types.TypedSymbol('i', 'int32')
    num_steps = pystencils.data_types.TypedSymbol('num_steps', 'int32')

    # step = step_size / projection_vector_norm
    # tex_coord = ray_equations.subs({t: min_t_tmp + i * step})
    tex_coord = ray_equations.subs({t: min_t_tmp}) + projection_vector * i

    assignments = {
        min_t_tmp: min_t,
        max_t_tmp: max_t,
        num_steps: sympy.ceiling((max_t_tmp - min_t_tmp) / (step_size / projection_vector_norm)),
        line_integral: sympy.Sum(volume_texture.at(tex_coord),
                                 (i, 0, num_steps)),
        intensity_weighting: (input_volume_field.coordinate_transform @ projection_vector).dot(central_ray) ** 2,
        output_projections_field.center(): (line_integral * step_size * intensity_weighting)
        # output_projections_field.center(): (max_t_tmp - min_t_tmp) / step # Uncomment to get path length
    }

    # def create_autodiff(self, constant_fields=None):
    # backward_assignments = backward_projection(AdjointField(output_projections_field),
    # AdjointField(input_volume_field),
    # projection_matrix,
    # 1)
    # self._autodiff = pystencils.autodiff.AutoDiffOp(
    # assignments, "op", constant_fields=constant_fields, backward_assignments=backward_assignments)

    # assignments._create_autodiff = types.MethodType(create_autodiff, assignments)

    return assignments


@crazy
def backward_projection(input_projection, output_volume, projection_matrix, normalization):
    projection_matrix = pystencils_reco.ProjectiveMatrix(projection_matrix)
    assignments = pystencils_reco.resampling.generic_spatial_matrix_transform(
        input_projection,
        output_volume,
        None,
        inverse_matrix=projection_matrix)

    for a in assignments.all_assignments:
        a = pystencils.Assignment(a.lhs, a.rhs / normalization)

    return assignments
    a = pystencils.Assignment(a.lhs, a.rhs / normalization)

    return assignments
