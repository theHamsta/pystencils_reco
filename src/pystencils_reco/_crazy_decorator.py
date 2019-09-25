# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import inspect
import types
from functools import partial

import sympy

import pycuda.gpuarray
import pystencils
from pystencils.field import Field

try:
    from pystencils.autodiff.backends._pytorch import torch_dtype_to_numpy
except Exception:
    pass

try:
    import torch
except ModuleNotFoundError:
    torch = None
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None
except ImportError:
    tf = None


class _WhatEverClass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _create_field_from_array_like(field_name, maybe_array):

    if torch:
        # Torch tensors don't have t.strides but t.stride(dim). Let's fix that!
        if isinstance(maybe_array, torch.Tensor):
            fake_array = _WhatEverClass(
                strides=[maybe_array.stride(i) for i in range(len(maybe_array.shape))],
                shape=maybe_array.shape,
                dtype=torch_dtype_to_numpy(maybe_array.dtype))
            field = Field.create_from_numpy_array(field_name, fake_array)
            return field

    return Field.create_from_numpy_array(field_name, maybe_array)


def coerce_to_field(field_name, array_like):
    if isinstance(array_like, Field):
        return array_like.new_field_with_different_name(field_name, array_like)
    return _create_field_from_array_like(field_name, array_like)


def is_array_like(a):
    return (hasattr(a, '__array__') or isinstance(a, pycuda.gpuarray.GPUArray)) and not isinstance(a, sympy.Matrix)


def crazy(function):

    def wrapper(*args, **kwargs):
        arg_names = inspect.getfullargspec(function).args
        compile_args = [_create_field_from_array_like(arg_names[i], a)
                        if is_array_like(a)
                        else a for i, a in enumerate(args)]
        compile_kwargs = {k: _create_field_from_array_like(str(k), a)
                          if (hasattr(a, '__array__') or isinstance(a, pycuda.gpuarray.GPUArray)) and
                          not isinstance(a, sympy.Matrix)  # noqa
                          else a for (k, a) in kwargs.items()}
        # compile_kwargs['function_name'] = function.__name__

        assignments = function(*compile_args, **compile_kwargs)

        kwargs.update({arg_names[i]: a for i, a in enumerate(args)})

        assignments.kwargs = kwargs

        if isinstance(assignments, pystencils.cpu.cpujit.KernelWrapper):
            assignments.code = str(pystencils.show_code(assignments.ast))
        if isinstance(assignments, types.FunctionType):
            if hasattr(assignments, 'code'):
                code = assignments.code
            else:
                code = ''
            assignments = partial(assignments, **kwargs)
            assignments.code = code

        return assignments

        # if isinstance(assignments, pystencils.AssignmentCollection):
        # assignments = pystencils_reco.AssignmentCollection(assignments)
        # is_gpu = any(isinstance(a, pycuda.gpuarray.GPUArray) for a in chain(args, kwargs.values()))
        # TODO: make accept: function_name=function.__name__
        # kernel = assignments.compile(target='gpu' if is_gpu else 'cpu')
        # else:
        # kernel = assignments
        # kernel_with_args = partial(kernel, *args, **kwargs)
        # if hasattr(kernel, 'code'):
        # kernel_with_args.code = kernel.code
        # kernel_with_args.assignments = assignments
        # return kernel_with_args

    return wrapper
