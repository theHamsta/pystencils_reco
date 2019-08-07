# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

# import itertools


# Would be way cooler if sympy.geometry supported Polygons in 3D 😞
# Let's use this SymPy fork then
# from diofant.geometry import Line, Point, Polygon, convex_hull, intersection


# def get_field_box_corner_points(field):
# """get_field_box

# :param field:
# """

# spatial_shape = field.spatial_shape

# corner_points = itertools.product(*[(0, s-1) for s in spatial_shape])
# corner_points = [field.coordinate_origin + field.coordinate_transform @ sympy.Matrix(p) for p in corner_points]
# corner_points = [Point(p) for p in corner_points]

# return corner_points


# def get_field_box_boundary(field):
# """get_field_box

# :param field:
# """

# spatial_shape = field.spatial_shape
# corner_points = list(itertools.product(*[(0, s-1) for s in diofant.symbols('x,y,z')]))

# polygons = []
# for i in range(field.spatial_dimensions):
# polygons.append(Polygon(p for p in corner_points if p[i] == 0))
# polygons.append(Polygon(p for p in corner_points if p[i] == spatial_shape[i]-1))

# # for i, p in enumerate(polygons):
# # polygons[i] = Polygon(Point(field.coordinate_origin + field.coordinate_transform @
# # sympy.Matrix(point)) for point in p.args)

# return polygons


# def calc_box_ray_intersections(field, field_equations, coordinate_symbols):
# """get_field_box

# :param field:
# """

# line = Line(ray_equations[s].subs({t: 0}) for s in coordinate_symbols)
# spatial_shape = field.spatial_shape

# corner_points = itertools.product(*[(0, s-1) for s in spatial_shape])
# corner_points = [field.coordinate_origin + field.coordinate_transform @ sympy.Matrix(p) for p in corner_points]
# corner_points = [Point(p) for p in corner_points]

# return corner_points

import sympy


def coordinate_in_field_conditions(field, coordinate_symbols, eps=1e-3):
    """get_field_box

    :param field:
    """
    coordinate_vector = sympy.Matrix(coordinate_symbols)
    # box_intervals = [sympy.Interval(0, s-1) for s in field.spatial_shape]

    # return list(coordinate in interval for coordinate, interval in zip(coordinate_vector, box_intervals))
    return list(c >= -eps for c in coordinate_vector) + \
        list(c <= shape+eps for c, shape in zip(coordinate_vector, field.spatial_shape))
