# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import numpy as np

import pystencils


class NumpyArrayHandler:

    def zeros(self, shape, dtype=np.float64, order='C'):
        return np.zeros(shape, dtype, order)

    def ones(self, shape, dtype, order='C'):
        return np.ones(shape, dtype, order)

    def empty(self, shape, dtype=np.float64, layout=None):
        if layout:
            cpu_array = pystencils.field.create_numpy_array_with_layout(shape, dtype, layout)
            return self.from_numpy(cpu_array)
        else:
            return np.empty(shape, dtype)

    def empty_like(self, array):
        return self.empty(array.shape, array.dtype)

    def ones_like(self, array):
        return self.ones(array.shape, array.dtype)

    def zeros_like(self, array):
        return self.ones(array.shape, array.dtype)

    def to_gpu(self, array):
        return array

    def upload(self, gpuarray, numpy_array):
        gpuarray[...] = numpy_array

    def download(self, gpuarray, numpy_array):
        numpy_array[...] = gpuarray

    def randn(self, shape, dtype=np.float64):
        return np.random.randn(*shape).astype(dtype)
