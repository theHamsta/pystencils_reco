# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""
Implements common image processing filters for arbitrary dimensionality
"""

import functools

import pystencils
import pystencils.simp
import pystencils_reco.functions


def mean_filter(input_field: pystencils.Field, output_field: pystencils.Field, stencil):
    assignments = pystencils.AssignmentCollection({
        output_field.center(): sum(input_field[t] for t in stencil) / len(stencil)
    }, {})

    return pystencils.simp.sympy_cse(assignments)


def gauss_filter(input_field: pystencils.Field, output_field: pystencils.Field, stencil, sigma):

    weighting_function = functools.partial(pystencils_reco.functions.gaussian, sigma=sigma)
    return generic_stationary_filter(input_field, output_field, stencil, weighting_function)


def bilateral_filter(input_field: pystencils.Field,
                     output_field: pystencils.Field,
                     stencil,
                     distance_sigma,
                     value_sigma):

    def weighting_function(offset, offset_value, central_value):
        distance_weight = pystencils_reco.functions.gaussian(offset, distance_sigma)
        value_weight = pystencils_reco.functions.gaussian((offset_value-central_value,), value_sigma)
        return value_weight * distance_weight

    return generic_instationary_filter(input_field, output_field, stencil, weighting_function)


def guided_bilateral_filter(input_field: pystencils.Field,
                            output_field: pystencils.Field,
                            stencil,
                            distance_sigma,
                            value_sigma):

    def weighting_function(offset, offset_value, guided_offset_value, central_value):
        distance_weight = pystencils_reco.functions.gaussian(offset, distance_sigma)
        value_weight = pystencils_reco.functions.gaussian((guided_offset_value-central_value,), value_sigma)
        return value_weight * distance_weight

    return generic_guided_filter(input_field, output_field, stencil, weighting_function)


def generic_stationary_filter(input_field: pystencils.Field,
                              output_field: pystencils.Field,
                              stencil,
                              weighting_function,
                              normalize_weights=True):
    """generic_function_filter

    :param input_field: 
    :type input_field: pystencils.Field
    :param output_field:
    :type output_field: pystencils.Field
    :param stencil:
    :param weighting_function: A function that takes a offset tuple and transfers it to weighting of the function value
    :param normalize_weights: whether or not to normalize weights to a sum of one
    """

    weights = 0
    sum = 0
    for s in stencil:
        weight = weighting_function(s)
        weights += weight
        sum += weight * input_field[s]

    assignments = pystencils.AssignmentCollection({
        output_field.center(): sum / weights if normalize_weights else sum
    }, {})

    return pystencils.simp.sympy_cse(assignments)


def generic_instationary_filter(input_field: pystencils.Field,
                                output_field: pystencils.Field,
                                stencil,
                                weighting_function,
                                normalize_weights=True):
    """Implements a generic instationary filter. The filter weight depends on the current stencil offset, the function value there and the central function value

    :param input_field: 
    :type input_field: pystencils.Field
    :param output_field:
    :type output_field: pystencils.Field
    :param stencil:
    :param weighting_function: A function that takes current offset, offset function value and stencils center function value
    :param normalize_weights: whether or not to normalize weights to a sum of one
    """

    weights = 0
    sum = 0
    for s in stencil:
        weight = weighting_function(s, input_field[s], input_field.center())
        weights += weight
        sum += weight * input_field[s]

    assignments = pystencils.AssignmentCollection({
        output_field.center(): sum / weights if normalize_weights else sum
    }, {})

    return pystencils.simp.sympy_cse(assignments)


def generic_guided_filter(input_field: pystencils.Field,
                          guide_field: pystencils.Field,
                          output_field: pystencils.Field,
                          stencil,
                          weighting_function,
                          normalize_weights=True):
    """Implements a generic non-stationary filter. The filter weight depends on the current stencil offset, the function value there and the central function value

    :param input_field: 
    :type input_field: pystencils.Field
    :param output_field:
    :type output_field: pystencils.Field
    :param stencil:
    :param weighting_function: A function that takes current offset, offset function value and stencils center function value
    :param normalize_weights: whether or not to normalize weights to a sum of one
    """

    weights = 0
    sum = 0
    for s in stencil:
        weight = weighting_function(s, input_field[s], guide_field[s], input_field.center())
        weights += weight
        sum += weight * input_field[s]

    assignments = pystencils.AssignmentCollection({
        output_field.center(): sum / weights if normalize_weights else sum
    }, {})

    return pystencils.simp.sympy_cse(assignments)
