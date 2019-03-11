# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import pystencils.assignment_collection
import pystencils.autodiff


class AssignmentCollection(pystencils.AssignmentCollection):
    """
    A high-level wrapper around pystencils.AssignmentCollection that provides some convenience methods
    for simpler usage in the field of image/volume processing
    """

    def __init__(self, arg):
        super(AssignmentCollection, self).__init__()
        self.arg = arg
        self._autodiff = None

    def compile(self, target='cpu', *args, **kwargs):
        """Convenience wrapper for pystencils.create_kernel(...).compile()
        See :func: ~pystencils.create_kernel
        """

        cse = pystencils.simp.sympy_cse(self)
        return pystencils.create_kernel(cse, target, *args, **kwargs).compile()

    def create_pytorch_op(self, **field_name_kwargs):
        if not self._autodiff:
            self._autodiff = pystencils.autodiff.AutoDiffOp(self, operation_name="", diff_mode='transposed-forward')
        input_field_to_tensor_map = {f: field_name_kwargs[f.name] for f in self._autodiff.forward_fields}

        return self._autodiff.create_tensorflow_op(input_field_to_tensor_map, backend='torch')
