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
from pystencils.cache import disk_cache, disk_cache_no_fallback
from pystencils_autodiff.field_tensor_conversion import ArrayWrapper


def crazy(function) -> pystencils_reco.AssignmentCollection:
    from pystencils_autodiff.field_tensor_conversion import (
        create_field_from_array_like, is_array_like)

    @functools.wraps(function)
    @disk_cache_no_fallback
    def wrapper(*args, **kwargs):
        import pycuda.gpuarray
        inspection = inspect.getfullargspec(function)
        arg_names = inspection.args
        annotations = inspection.annotations

        compile_args = {arg_names[i]: create_field_from_array_like(arg_names[i], a, annotations.get(arg_names[i], None))
                        if is_array_like(a)
                        else a for i, a in enumerate(args)}
        compile_kwargs = {k: create_field_from_array_like(str(k), a, annotations.get(k, None))
                          if (hasattr(a, '__array__') or isinstance(a, pycuda.gpuarray.GPUArray)) and
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


# class requires(object):
# """ Decorator for registering requirements on print methods. """

# def __init__(self, **kwargs):
# self._decorator_kwargs = kwargs

# def __call__(self, function):
# def _method_wrapper(self, *args, **kwargs):
# for k, v in self._decorator_kwargs.items():
# obj, member = k.split('__')
# setattr(kwargs[obj], member, v)

# return function(*args, **kwargs)
# return functools.wraps(function)(_method_wrapper)
