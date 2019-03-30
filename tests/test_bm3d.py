# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

try:
    import pyconrad.autoinit
except:  # NOQA
    import unittest.mock
    pyconrad = unittest.mock.MagicMock()

from os.path import dirname, join

import numpy as np
import pycuda.autoinit  # noqa
import skimage.io
from pycuda.gpuarray import to_gpu, zeros, zeros_like

from pystencils_reco.bm3d import Bm3d
from pystencils_reco.stencils import BallStencil, BoxStencil


def test_bm3d():
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
                BoxStencil(4, ndim),
                BallStencil(2, ndim),
                compilation_target='gpu',
                max_block_matches=8,
                threshold=0.1)

    block_scores = zeros(bm3d.block_scores.shape, bm3d.block_scores.dtype.numpy_dtype)
    block_matched = zeros(bm3d.block_matched_field.shape, bm3d.block_matched_field.dtype.numpy_dtype)

    print(bm3d.block_matching.code)
    print(bm3d.aggregate.code)
    print(bm3d.collect_patches.code)

    bm3d.block_matching(block_scores=block_scores)
    pyconrad.imshow(block_scores, 'block_scores')

    bm3d.collect_patches(block_scores=block_scores, block_matched=block_matched)
    pyconrad.imshow(block_matched, 'block_matched')

    bm3d.aggregate(block_scores=block_scores, block_matched=block_matched)
    pyconrad.imshow(lenna_noisy, 'denoised')


def main():
    test_bm3d()


if __name__ == '__main__':
    main()
