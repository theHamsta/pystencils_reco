# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""
Implements common image processing filters for arbitrary dimensionality
"""

import itertools

import pystencils
import pystencils.simp
import pystencils_reco.functions


def mean_filter(input_field: pystencils.Field, output_field: pystencils.Field, stencil):
    assignments = pystencils.AssignmentCollection({
        output_field.center(): sum(input_field[t] for t in stencil) / len(stencil)
    }, {})

    return pystencils.simp.sympy_cse(assignments)


def gauss_filter(input_field: pystencils.Field, output_field: pystencils.Field, stencil, sigma):

    weights = 0
    sum = 0
    for s in stencil:
        weight = pystencils_reco.functions.gaussian(s, sigma)
        weights += weight
        sum += weight * input_field[s]

    assignments = pystencils.AssignmentCollection({
        output_field.center(): sum / weights
    }, {})

    return pystencils.simp.sympy_cse(assignments)


def generic_stationary_filter(input_field: pystencils.Field,
                              output_field: pystencils.Field,
                              stencil,
                              weighting_function):
    """generic_function_filter

    :param input_field: 
    :type input_field: pystencils.Field
    :param output_field:
    :type output_field: pystencils.Field
    :param stencil:
    :param weighting_function: A function that takes a offset tuple and transfers it to weighting of the function value
    """

    weights = 0
    sum = 0
    for s in stencil:
        weight = weighting_function(s)
        weights += weight
        sum += weight * input_field[s]

    assignments = pystencils.AssignmentCollection({
        output_field.center(): sum / weights
    }, {})

    return pystencils.simp.sympy_cse(assignments)
