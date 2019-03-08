# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""
Implements common stencil types, like :func: ~pystencils_reco.stencils.BoxStencil
and :func: ~pystencils_reco.stencils.BallStencil
"""
import itertools
import math
from typing import Union

import pampy

from pystencils.stencils import *  # NOQA


class Stencil(list):
    """Implements an Iterable of relative accesses on a Field"""

    def __init__(self, iterable, ndim):
        assert (len(i) == ndim for i in iterable), "All elements of a Stencils should have len == ndim"
        super(Stencil, self).__init__(list(iterable))
        self.ndim = ndim


class LineStencil(Stencil):
    """Stencil along one dimension. Results in a 1D filter"""

    def __init__(self, kernel_size: int, filter_dimension, ndim=3, with_center=True):
        stencil = [0] * ndim
        stencil[filter_dimension] = range(-(kernel_size//2), kernel_size//2 + 1)

        if not with_center:
            stencil.remove(tuple([0]*ndim))
        super(LineStencil, self).__init__(stencil, ndim)


class BoxStencil(Stencil):
    """Implements a rectangular stencil"""

    def __init__(self, kernel_size: Union[int, tuple], ndim=3, with_center=True):
        stencil = pampy.match(kernel_size,
                              int, lambda _: itertools.product(
                                  range(-(kernel_size//2), kernel_size//2 + 1), repeat=ndim),
                              pampy.ANY, lambda _: itertools.product(
                                  *[range(-(i//2), i//2 + 1) for i in kernel_size])
                              )
        if not with_center:
            stencil = list(stencil)
            stencil.remove(tuple([0]*ndim))

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * ndim
        else:
            ndim = len(kernel_size)
        assert (i % 2 == 1 for i in kernel_size), "kernel_size must consist of uneven numbers"

        super(BoxStencil, self).__init__(stencil, ndim)


class BallStencil(Stencil):
    """BallStencil"""

    def __init__(self, radius, ndim=3):

        stencil = []
        circumscribing_box = BoxStencil(2*radius+1, ndim)
        for s in circumscribing_box:
            norm = math.sqrt(sum(i*i for i in s))
            if norm <= radius:
                stencil.append(s)

        super(BallStencil, self).__init__(stencil, ndim)
