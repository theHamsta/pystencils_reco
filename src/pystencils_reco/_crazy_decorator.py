# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

from pystencils.field import create_from_numpy_array

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


def crazy(function):

    def wrapper(*args, **kwargs):
        compile_args = [create_from_numpy_array(a) if hasattr(a, '__array__') else a for a in args]
        compile_kwargs = {k: create_from_numpy_array(a) if hasattr(
            a, '__array__') else a for k, a in zip(kwargs.items())}

        assignments = function(*compile_args, **compile_kwargs)

        return assignments
        # if torch:
        # is_torch = any(isinstance(a, torch.Tensor) for a in chain(args, kwargs.values()))
        # if is_torch:
        # pass

    return wrapper
