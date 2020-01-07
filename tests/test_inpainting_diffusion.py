# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""


import numpy as np
import pyconrad.autoinit
import pycuda.gpuarray as gpuarray
import pytest
import sympy
from pyconrad.phantoms import shepp_logan
from tqdm import trange

import pystencils


def inpainting_diffusion(defect_image, mask, diffusion_coefficient=0.02, num_iterations=2000):

    inpainted = np.copy(defect_image).astype(np.float32)
    c_arr = gpuarray.zeros(defect_image.shape, np.float32)
    c_next_arr = gpuarray.zeros_like(c_arr)
    mask_arr = gpuarray.zeros(mask.shape, mask.dtype)
    c, c_next, mask_field = pystencils.fields(
        "c, c_next, mask_field", c=c_arr, c_next=c_next_arr, mask_field=mask_arr)
    adv_diff_pde = pystencils.fd.transient(c) - pystencils.fd.diffusion(c,
                                                                        sympy.Symbol("D"))  # + pystencils.fd.advection(c, v)

    discretize = pystencils.fd.Discretization2ndOrder(1, diffusion_coefficient)
    discretization = discretize(adv_diff_pde)

    @pystencils.kernel
    def kernel():
        c_next.center @= discretization.subs(
            sympy.Symbol("D"), 1) if mask_field.center > 0.5 else c.center

    kernel = pystencils.create_kernel(kernel, target='gpu').compile()

    c_arr.set(inpainted)
    mask_arr.set(mask)
    for _ in trange(num_iterations, desc='Inpainting', leave=True):
        kernel(c=c_arr, c_next=c_next_arr, mask_field=mask_arr)
        c_arr, c_next_arr = c_next_arr, c_arr
        # pyconrad.imshow(c_next_arr[len(c_next_arr) // 2])

    c_arr.get(inpainted)

    return inpainted


def test_inpainting_diffusion():
    pytest.importorskip("pycuda")
    import pycuda.autoinit  # noqa
    shape = (100, 100, 100)

    phantom = shepp_logan(*shape)
    mask = np.random.rand(*shape) > 0.5

    phantom[mask] = 0
    pyconrad.imshow(phantom, '1')

    inpainted = inpainting_diffusion(phantom, mask, num_iterations=3000)

    pyconrad.imshow(inpainted, '2')
