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
from typing import Tuple, Union


class Stencil(list):
    """Implements an Iterable of relative accesses on a Field"""

    def __init__(self, iterable, ndim, with_center=True):
        assert (len(i) == ndim for i in iterable), "All elements of a Stencil should have len == ndim"
        super(Stencil, self).__init__(list(iterable))
        self.ndim = ndim

        if not with_center:
            try:
                self.remove(tuple([0]*ndim))
            except ValueError:
                pass

    def as_strided(self, strides: Tuple):
        """
        Converts a regular Stencil into a strided stencil by omitting
        every i,j,n-th element along the different dimensions
        """
        new_stencil = [s for s in self if all(s[i] % t == 0 for i, t in enumerate(strides))]
        return Stencil(new_stencil, self.ndim)

    @property
    def shape(self):
        mins = [min(s[i] for s in self) for i in range(self.ndim)]
        maxs = [max(s[i] for s in self) for i in range(self.ndim)]
        return tuple(a - b + 1 for a, b in zip(maxs, mins))


class LineStencil(Stencil):
    """Stencil along one dimension. Results in a 1D filter"""

    def __init__(self, kernel_size: int, filter_dimension, ndim=3, with_center=True):
        stencil = []
        for offset in range(-(kernel_size//2), -(kernel_size//2) + kernel_size):
            stencils_element = [0] * ndim
            stencils_element[filter_dimension] = offset
            stencil.append(tuple(stencils_element))

        super(LineStencil, self).__init__(stencil, ndim, with_center)


class BoxStencil(Stencil):
    """Implements a rectangular stencil"""

    def __init__(self, kernel_size: Union[int, tuple], ndim=3, with_center=True):
        if isinstance(kernel_size, int):
            stencil = itertools.product(range(-(kernel_size // 2), -(kernel_size // 2) + kernel_size), repeat=ndim)
        else:
            stencil = itertools.product(*[range(-(i // 2), -(i // 2) + i) for i in kernel_size])

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * ndim
        else:
            ndim = len(kernel_size)

        super(BoxStencil, self).__init__(stencil, ndim, with_center)


class BallStencil(Stencil):
    """
    A spherical stencil that includes all relative accesses with
    distance smaller than a given radius to the stencil center
    """

    def __init__(self, radius, ndim=3, with_center=True):
        self.radius = radius

        stencil = []
        circumscribing_box = BoxStencil(2 * radius + 1, ndim)
        for s in circumscribing_box:
            norm = math.sqrt(sum(i * i for i in s))
            if norm <= radius:
                stencil.append(s)

        super(BallStencil, self).__init__(stencil, ndim, with_center)
