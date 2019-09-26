# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

from functools import partial

import pystencils.assignment_collection

# TODO: find good name to differentiate from conventional pystencils.AssignmentCollection... Perhaps ImageFilter?


class AssignmentCollection(pystencils.AssignmentCollection):
    """
    A high-level wrapper around pystencils.AssignmentCollection that provides some convenience methods
    for simpler usage in the field of image/volume processing.

    Better defaults for Image Processing.
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
        self._autodiff = None

    def compile(self, target='cpu', *args, **kwargs):
        """Convenience wrapper for pystencils.create_kernel(...).compile()
        See :func: ~pystencils.create_kernel
        """

        if 'data_type' not in kwargs:
            kwargs['data_type'] = 'float32'

        if 'cpu_openmp' not in kwargs:
            kwargs['cpu_openmp'] = True

        ast = pystencils.create_kernel(self, target, *args, **kwargs)
        code = pystencils.show_code(ast)
        kernel = ast.compile()
        if hasattr(self, 'args'):
            kernel = partial(kernel, *self.args)

        if hasattr(self, 'kwargs'):
            kernel = partial(kernel, **self.kwargs)

        kernel.code = code
        return kernel

    def backward(self):
        if not self._autodiff:
            self._create_autodiff()
        return AssignmentCollection(self._autodiff.backward_assignments)

    def create_pytorch_op(self, target='gpu', **kwargs):
        return self._create_ml_op('torch_native', target, **kwargs)

    def create_tensorflow_op(self, target='gpu', **kwargs):
        return self._create_ml_op('tensorflow_native', target, **kwargs)

    def _create_ml_op(self, backend, target, **kwargs):
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
            self, operation_name="", constant_fields=constant_fields)
