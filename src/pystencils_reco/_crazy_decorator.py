# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import inspect

import pystencils
import pystencils_reco
import sympy


def crazy(function):
    from pystencils_autodiff.field_tensor_conversion import (
        create_field_from_array_like, is_array_like)

    def wrapper(*args, **kwargs):
        import pycuda.gpuarray
        arg_names = inspect.getfullargspec(function).args
        compile_args = [create_field_from_array_like(arg_names[i], a)
                        if is_array_like(a)
                        else a for i, a in enumerate(args)]
        compile_kwargs = {k: create_field_from_array_like(str(k), a)
                          if (hasattr(a, '__array__') or isinstance(a, pycuda.gpuarray.GPUArray)) and
                          not isinstance(a, sympy.Matrix)  # noqa
                          else a for (k, a) in kwargs.items()}
        # compile_kwargs['function_name'] = function.__name__

        assignments = function(*compile_args, **compile_kwargs)

        kwargs.update({arg_names[i]: a for i, a in enumerate(args)})

        if (isinstance(assignments, pystencils.AssignmentCollection) and
                not isinstance(assignments, pystencils_reco.AssignmentCollection)):
            assignments = pystencils_reco.AssignmentCollection(assignments)

        try:
            assignments.kwargs = kwargs
        except Exception:
            pass

        return assignments

    return wrapper
