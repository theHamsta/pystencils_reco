# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

from enum import Enum
from functools import partial
from itertools import chain

import pystencils.assignment_collection


class NdArrayType(str, Enum):
    UNKNOWN = None
    TORCH = 'torch'
    TENSORFLOW = 'tensorflow'
    NUMPY = 'numpy'
    PYCUDA = 'pycuda'


def get_type_of_arrays(*args):
    try:
        from pycuda.gpuarray import GPUArray
        if any(isinstance(a, GPUArray) for a in args):
            return NdArrayType.PYCUDA
    except Exception:
        pass
    try:
        import torch
        if any(isinstance(a, torch.Tensor) for a in args):
            return NdArrayType.TORCH
    except Exception as e:
        print(e)
    try:
        from tensorflow import Tensor
        if any(isinstance(a, Tensor) for a in args):
            return NdArrayType.TENSORFLOW
    except Exception:
        pass
    try:
        if any(hasattr(a, '__array__') for a in args):
            return NdArrayType.NUMPY
    except Exception:
        pass

    return NdArrayType.UNKNOWN


class AssignmentCollection(pystencils.AssignmentCollection):
    """
    A high-level wrapper around pystencils.AssignmentCollection that provides some convenience methods
    for simpler usage in the field of image/volume processing.

    Better defaults for Image Processing.

    .. todo::
        find good name to differentiate from conventional pystencils.AssignmentCollection... Perhaps ImageFilter?
    """

    def __init__(self, assignments, perform_cse=True, *args, **kwargs):
        if isinstance(assignments, pystencils.AssignmentCollection):
            assignments = assignments.all_assignments

        assignments = pystencils.AssignmentCollection(assignments, {})
        if perform_cse:
            main_assignments = [a for a in assignments if isinstance(a.lhs, pystencils.Field.Access)]
            subexpressions = [a for a in assignments if not isinstance(a.lhs, pystencils.Field.Access)]
            assignments = pystencils.AssignmentCollection(main_assignments, subexpressions)
            assignments = pystencils.simp.sympy_cse(assignments)
        super(AssignmentCollection, self).__init__(assignments.all_assignments, {}, *args, **kwargs)
        self.args = []
        self.kwargs = {}
        self._autodiff = None
        self.kernel = None

    def compile(self, target=None, *args, **kwargs):
        """Convenience wrapper for pystencils.create_kernel(...).compile()
        See :func: ~pystencils.create_kernel
        """

        if 'data_type' not in kwargs:
            kwargs['data_type'] = 'float32'

        if 'cpu_openmp' not in kwargs:
            kwargs['cpu_openmp'] = True

        array_type = get_type_of_arrays(*chain(args, self.args, self.kwargs.values(), kwargs.values()))

        if array_type == NdArrayType.TORCH:
            kernel = self._create_ml_op('torch_native', target, **kwargs)
        elif array_type == NdArrayType.TENSORFLOW:
            kernel = self._create_ml_op('tensorflow_native', target, **kwargs)
        else:
            if array_type == NdArrayType.PYCUDA:
                target = 'gpu'
            if not target:
                target = 'cpu'
            ast = pystencils.create_kernel(self,
                                           target=target,
                                           *args,
                                           **kwargs)
            kernel = ast.compile()

        if self.args:
            kernel.__call__ = partial(kernel, *self.args)

        if self.kwargs:
            kernel = partial(kernel, **self.kwargs)

        self.kernel = kernel
        return kernel

    def backward(self):
        if not self._autodiff:
            self._create_autodiff()
        return AssignmentCollection(self._autodiff.backward_assignments)

    @property
    def code(self):
        return self.kernel.code

    def create_pytorch_op(self, target='gpu', **kwargs):
        return self._create_ml_op('torch_native', target, **kwargs)

    def create_tensorflow_op(self, target='gpu', **kwargs):
        return self._create_ml_op('tensorflow_native', target, **kwargs)

    def _create_ml_op(self, backend, target, **kwargs):
        if not target:
            target = 'gpu'
        constant_field_names = [f for f, t in kwargs.items()
                                if hasattr(t, 'requires_grad') and not t.requires_grad]
        constant_fields = {f for f in self.free_fields if f.name in constant_field_names}

        if not self._autodiff:
            self._create_autodiff(constant_fields)

        op = self._autodiff.create_tensorflow_op(backend=backend, use_cuda=(target == 'gpu'))

        if hasattr(self, 'args'):
            op = partial(op, *self.args)

        if hasattr(self, 'kwargs'):
            op = partial(op, **self.kwargs)

        return op

    def _create_autodiff(self, constant_fields=[]):
        import pystencils.autodiff
        self._autodiff = pystencils.autodiff.AutoDiffOp(
            self, constant_fields=constant_fields)
