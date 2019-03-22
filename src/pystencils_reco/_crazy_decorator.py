# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import inspect
from uuid import uuid4

from pystencils.field import Field

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
        arg_names = inspect.getfullargspec(function).args
        compile_args = [Field.create_from_numpy_array(
            arg_names[i], a) if hasattr(a, '__array__') else a for i, a in enumerate(args)]
        compile_kwargs = {k: Field.create_from_numpy_array(str(k), a) if hasattr(
            a, '__array__') else a for (k, a) in kwargs.items()}

        assignments = function(*compile_args, **compile_kwargs)

        return assignments
        # if torch:
        # is_torch = any(isinstance(a, torch.Tensor) for a in chain(args, kwargs.values()))
        # if is_torch:
        # pass

    return wrapper
