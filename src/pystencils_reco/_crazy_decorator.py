# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import inspect

import sympy

from pystencils.autodiff.backends._pytorch import torch_dtype_to_numpy
from pystencils.field import Field

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


def crazy(function):

    def wrapper(*args, **kwargs):
        arg_names = inspect.getfullargspec(function).args
        compile_args = [_create_field_from_array_like(arg_names[i], a)
                        if hasattr(a, '__array__') and not isinstance(a, sympy.Matrix)
                        else a for i, a in enumerate(args)]
        compile_kwargs = {k: _create_field_from_array_like(str(k), a) if hasattr(
            a, '__array__') else a for (k, a) in kwargs.items()}
        # compile_kwargs['function_name'] = function.__name__

        assignments = function(*compile_args, **compile_kwargs)

        if torch:
            is_torch = all(isinstance(a, torch.Tensor) if hasattr(a, '__array__')
                           or isinstance(a, Field) else a for a in args)
            if is_torch:
                kwargs.update({arg_names[i]: a for i, a in enumerate(args)})
                return assignments.create_pytorch_op(**kwargs)

        return assignments

    return wrapper
