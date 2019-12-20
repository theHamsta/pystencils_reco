# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import sympy

import pystencils


def inpainting_diffusion(input_field, output_field, mask, diffusion_coefficient=0.2):

    adv_diff_pde = pystencils.fd.transient(input_field) - pystencils.fd.diffusion(input_field,
                                                                                  sympy.Symbol("D"))
    #                                                                             + pystencils.fd.advection(c, v)

    discretize = pystencils.fd.Discretization2ndOrder(1, diffusion_coefficient)
    discretization = discretize(adv_diff_pde)

    @pystencils.kernel
    def kernel():
        output_field.center @= discretization.subs(sympy.Symbol("D"), 1) if mask.center > 0.5 else input_field.center
