# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import functools
import inspect

import sympy

import pystencils
import pystencils_autodiff.transformations
import pystencils_reco
from pystencils.cache import disk_cache
from pystencils_autodiff.field_tensor_conversion import ArrayWrapper


def crazy(function) -> pystencils_reco.AssignmentCollection:
    from pystencils_autodiff.field_tensor_conversion import create_field_from_array_like, is_array_like

    # @disk_cache_no_fallback
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        inspection = inspect.getfullargspec(function)
        arg_names = inspection.args
        annotations = inspection.annotations

        compile_args = {arg_names[i]: create_field_from_array_like(arg_names[i], a, annotations.get(arg_names[i], None))
                        if is_array_like(a)
                        else a for i, a in enumerate(args)}
        compile_kwargs = {k: create_field_from_array_like(str(k), a, annotations.get(k, None))
                          if (hasattr(a, '__array__') or 'GPUArray' in str(a.__class__)) and
                          not isinstance(a, sympy.Matrix)  # noqa
                          else a for (k, a) in kwargs.items()}

        try:
            assignments = disk_cache(function)(*compile_args.values(), **compile_kwargs)
        except Exception:
            assignments = function(*compile_args.values(), **compile_kwargs)

        kwargs.update({arg_names[i]: a for i, a in enumerate(args)})

        if (isinstance(assignments, (pystencils.AssignmentCollection, list, dict)) and
                not isinstance(assignments, pystencils_reco.AssignmentCollection)):
            assignments = pystencils_reco.AssignmentCollection(assignments)

        kwargs = {k: v if not isinstance(v, ArrayWrapper) else v.array for k, v in kwargs.items()}

        try:
            assignments.kwargs = kwargs
        except Exception:
            pass

        return assignments

    return wrapper


def fixed_boundary_handling(function):

    @functools.wraps(function)
    def wrapper(*args, **kwargs):

        assignments = function(*args, **kwargs)
        kwargs = assignments.__dict__.get('kwargs', {})
        args = assignments.__dict__.get('args', {})

        if (isinstance(assignments, (pystencils.AssignmentCollection, list, dict)) and
                not isinstance(assignments, pystencils_reco.AssignmentCollection)):
            assignments = pystencils_reco.AssignmentCollection(assignments)

        assignments = pystencils_autodiff.transformations.add_fixed_constant_boundary_handling(assignments)

        assignments = pystencils_reco.AssignmentCollection(assignments)
        assignments.args = args
        assignments.kwargs = kwargs

        return assignments

    return wrapper


@disk_cache
def crazy_compile(crazy_function, *args, **kwargs):

    return crazy_function(*args, **kwargs).compile()
