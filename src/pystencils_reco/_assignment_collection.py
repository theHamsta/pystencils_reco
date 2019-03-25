# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import pystencils.assignment_collection
import pystencils.autodiff


# TODO: find good name to differentiate from conventional pystencils.AssignmentCollection... Perhaps ImageFilter?
class AssignmentCollection(pystencils.AssignmentCollection):
    """
    A high-level wrapper around pystencils.AssignmentCollection that provides some convenience methods
    for simpler usage in the field of image/volume processing
    """

    def __init__(self, assignments, perform_cse=True, *args, **kwargs):
        if isinstance(assignments, pystencils.AssignmentCollection):
            assignments = assignments.all_assignments
        if perform_cse:
            assignments = pystencils.AssignmentCollection(assignments, {})
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

        return pystencils.create_kernel(self, target, *args, **kwargs).compile()

    def backward(self):
        if not self._autodiff:
            self._create_autodiff()
        return AssignmentCollection(self._autodiff.backward_assignments)

    def create_pytorch_op(self, **field_name_kwargs):
        input_field_to_tensor_map = {f: field_name_kwargs[f.name] for f in self.free_fields}
        constant_fields = [f for f, t in input_field_to_tensor_map.items() if not t.requires_grad]

        if not self._autodiff:
            self._create_autodiff(constant_fields)

        return self._autodiff.create_tensorflow_op(input_field_to_tensor_map, backend='torch')

    def _create_autodiff(self, constant_fields=[]):
        self._autodiff = pystencils.autodiff.AutoDiffOp(
            self, operation_name="", diff_mode='transposed-forward', constant_fields=constant_fields)
