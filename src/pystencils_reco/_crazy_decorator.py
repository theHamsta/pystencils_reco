# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import inspect

from pystencils.autodiff.backends._pytorch import torch_dtype_to_numpy
from pystencils.field import Field

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


def _create_field_from_array_like(field_name, maybe_array):

    if torch:
        # Torch tensors don't have t.strides but t.stride(dim). Let's fix that!
        if isinstance(maybe_array, torch.Tensor):
            fake_array = object()
            fake_array.strides = [maybe_array.stride(i) for i in range(len(maybe_array.shape))]
            fake_array.shape = maybe_array.shape
            fake_array.dtype = torch_dtype_to_numpy(maybe_array.dtype)
            field = Field.create_from_numpy_array(field_name, fake_array)
            return field

    return Field.create_from_numpy_array(field_name, maybe_array)


def crazy(function):

    def wrapper(*args, **kwargs):
        arg_names = inspect.getfullargspec(function).args
        compile_args = [_create_field_from_array_like(
            arg_names[i], a) if hasattr(a, '__array__') else a for i, a in enumerate(args)]
        compile_kwargs = {k: _create_field_from_array_like(str(k), a) if hasattr(
            a, '__array__') else a for (k, a) in kwargs.items()}

        assignments = function(*compile_args, **compile_kwargs)

        return assignments
        # if torch:
        # is_torch = any(isinstance(a, torch.Tensor) for a in chain(args, kwargs.values()))
        # if is_torch:
        # pass

    return wrapper
