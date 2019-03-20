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
        if perform_cse:
            assignments = pystencils.simp.sympy_cse(pystencils.AssignmentCollection(assignments, {})).all_assignments
        super(AssignmentCollection, self).__init__(assignments, {}, *args, **kwargs)
        self._autodiff = None

    def compile(self, target='cpu', *args, **kwargs):
        """Convenience wrapper for pystencils.create_kernel(...).compile()
        See :func: ~pystencils.create_kernel
        """
        if 'data_type' not in kwargs:
            kwargs['data_type'] = 'float'

        if 'cpu_openmp' not in kwargs:
            kwargs['cpu_openmp'] = True

        return pystencils.create_kernel(self, target, *args, **kwargs).compile()

    def backward(self):
        if not self._autodiff:
            self._create_autodiff()
        return self._autodiff.backward_assignments

    def create_pytorch_op(self, **field_name_kwargs):
        if not self._autodiff:
            self._create_autodiff()
        input_field_to_tensor_map = {f: field_name_kwargs[f.name] for f in self._autodiff.forward_fields}

        return self._autodiff.create_tensorflow_op(input_field_to_tensor_map, backend='torch')

    def _create_autodiff(self):
        self._autodiff = pystencils.autodiff.AutoDiffOp(self, operation_name="", diff_mode='transposed-forward')
