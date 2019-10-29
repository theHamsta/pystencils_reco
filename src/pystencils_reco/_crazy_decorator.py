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
import pystencils_reco
from pystencils_autodiff.field_tensor_conversion import ArrayWithIndexDimensions


def crazy(function):
    from pystencils_autodiff.field_tensor_conversion import (
        create_field_from_array_like, is_array_like)

    @functools.wraps(function)
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

        assignments = function(*compile_args.values(), **compile_kwargs)

        kwargs.update({arg_names[i]: a for i, a in enumerate(args)})

        if (isinstance(assignments, (pystencils.AssignmentCollection, list)) and
                not isinstance(assignments, pystencils_reco.AssignmentCollection)):
            assignments = pystencils_reco.AssignmentCollection(assignments)

        kwargs = {k: v if not isinstance(v, ArrayWithIndexDimensions) else v.array for k, v in kwargs.items()}

        try:
            assignments.kwargs = kwargs
        except Exception:
            pass

        return assignments

    return wrapper


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
