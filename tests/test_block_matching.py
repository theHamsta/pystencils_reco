# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from os.path import dirname, join

import numpy as np
import skimage.io
from tqdm import trange

import pystencils
from pystencils_reco.block_matching import block_matching_integer_offsets
from pystencils_reco.stencils import BallStencil


def test_block_matching():
    import pyconrad.autoinit

    block_stencil = BallStencil(5, ndim=2)
    matching_stencil = BallStencil(5, ndim=2)

    x, y, matches = pystencils.fields('x,y, matches(%i): float32[2d]' % len(matching_stencil))
    block_matching = block_matching_integer_offsets(x, y, matches, block_stencil, matching_stencil)
    print(block_matching)
    block_matching = block_matching.compile('cpu')

    test_image = 1-skimage.io.imread(join(dirname(__file__), "test_data",  "test_vessel2d_mask.png"), as_gray=True)
    test_image = np.ascontiguousarray(test_image, np.float32)
    result = np.zeros([*test_image.shape, len(matching_stencil)], np.float32)
    block_matching(x=test_image, y=test_image, matches=result)
    pyconrad.imshow(test_image, 'original')
    pyconrad.imshow(test_image, 'compare')
    pyconrad.imshow(np.swapaxes(result, 0, -1), 'result')


def test_block_matching_gpu():
    import pyconrad.autoinit
    import pycuda.autoinit
    from pycuda.gpuarray import to_gpu, zeros

    block_stencil = BallStencil(5, ndim=2)
    matching_stencil = BallStencil(5, ndim=2)

    x, y, matches = pystencils.fields('x,y, matches(%i): float32[2d]' % len(matching_stencil))
    block_matching = block_matching_integer_offsets(x, y, matches, block_stencil, matching_stencil)
    # print(block_matching)
    block_matching = block_matching.compile('gpu')

    test_image = 1-skimage.io.imread(join(dirname(__file__), "test_data",  "test_vessel2d_mask.png"), as_gray=True)
    test_image = to_gpu(np.ascontiguousarray(test_image, np.float32))
    result = zeros([*test_image.shape, len(matching_stencil)], np.float32)
    for i in trange(1000):
        block_matching(x=test_image, y=test_image, matches=result)
    pyconrad.imshow(test_image, 'original')
    pyconrad.imshow(test_image, 'compare')
    pyconrad.imshow(np.swapaxes(result.get(), 0, -1), 'result')


def main():
    # test_block_matching()
    test_block_matching_gpu()


if __name__ == '__main__':
    main()
