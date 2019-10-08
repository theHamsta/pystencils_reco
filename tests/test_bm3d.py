# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""


from os.path import dirname, join

import numpy as np
import pytest
import skimage.io

from pystencils_reco.bm3d import Bm3d
from pystencils_reco.stencils import BallStencil, BoxStencil


@pytest.mark.skip(reason="")
def test_bm3d():
    pytest.importorskip('pycuda')
    import pycuda.autoinit  # noqa
    try:
        import pyconrad.autoinit
    except:  # NOQA
        import unittest.mock
        pyconrad = unittest.mock.MagicMock()

    from pycuda.gpuarray import to_gpu, zeros
    lenna_file = join(dirname(__file__), "test_data", "lenna.png")
    lenna = skimage.io.imread(lenna_file, as_gray=True).astype(np.float32)

    lenna_noisy = lenna + np.random.rand(*lenna.shape).astype(np.float32)
    lenna_denoised = np.zeros_like(lenna)
    ndim = 2

    pyconrad.imshow(lenna, 'lenna')
    pyconrad.imshow(lenna_noisy, 'lenna_noisy')
    lenna = to_gpu(lenna)
    lenna_noisy = to_gpu(lenna_noisy)
    lenna_denoised = to_gpu(lenna_denoised)

    bm3d = Bm3d(lenna_noisy,
                lenna_denoised,
                BoxStencil(5, ndim),
                BallStencil(3, ndim),
                compilation_target='gpu',
                max_block_matches=8,
                blockmatching_threshold=1,
                hard_threshold=10,
                wiener_sigma=1)

    block_scores = zeros(bm3d.block_scores.shape, bm3d.block_scores.dtype.numpy_dtype)
    weights = zeros(bm3d.block_scores.shape, bm3d.block_scores.dtype.numpy_dtype)
    block_matched = zeros(bm3d.block_matched_field.shape, bm3d.block_matched_field.dtype.numpy_dtype)

    print(bm3d.block_matching.code)
    print(bm3d.aggregate.code)
    print(bm3d.collect_patches.code)

    bm3d.block_matching(block_scores=block_scores)
    pyconrad.imshow(block_scores, 'block_scores')

    bm3d.collect_patches(block_scores=block_scores, block_matched=block_matched)
    pyconrad.imshow(block_matched, 'block_matched')

    print(block_matched.shape)
    # fft[...] = block_matched

    # reikna.fft.FFT(fft, (-3, -2, -1))
    # pyconrad.imshow(fft, 'FFT')
    import skcuda.fft as cufft
    # forward_plan = cufft.cufftPlan3d(8, 4, 4, cufft.CUFFT_R2C)

    plan = cufft.Plan((8, 5, 5), np.complex64, np.complex64, 512*512)
    # forward_plan = cufft.cufftPlanMany(3, [8, 4, 4],
    # 0, 0, 0,
    # 0, 0, 0, cufft.CUFFT_R2C, 512*512)
    # backward_plan = cufft.cufftPlanMany(3, [8, 4, 4],
    # 0, 0, 0,
    # 0, 0, 0, cufft.CUFFT_C2R, 512*512)

    block_matched = block_matched.astype(np.complex64)
    reshaped = pycuda.gpuarray.reshape(block_matched, (512*512, 8, 5, 5))
    cufft.fft(reshaped, reshaped, plan)
    pyconrad.imshow(reshaped, 'FFT')
    print('fft')

    real_shaped = pycuda.gpuarray.reshape(block_matched.view(np.float32), bm3d.complex_field.shape)
    bm3d.hard_thresholding(complex_field=real_shaped, group_weights=weights)
    # bm3d.wiener_filtering(complex_field=real_shaped, group_weights=weights)

    cufft.ifft(reshaped, reshaped, plan, scale=True)
    pyconrad.imshow(reshaped, 'iFFT')
    print('ifft')

    reshaped = pycuda.gpuarray.reshape(reshaped.real, (512, 512, 8, 5*5))

    import pyconrad
    pyconrad.imshow(lenna_noisy, 'noisy')
    bm3d.aggregate(block_scores=block_scores, block_matched=reshaped, group_weights=weights, accumulated_weights=lenna)
    lenna_denoised /= lenna
    pyconrad.imshow(lenna_denoised, 'denoised')
