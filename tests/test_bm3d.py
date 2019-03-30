# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""

try:
    import pyconrad.autoinit
    raise NotImplementedError()
except:  # NOQA
    import unittest.mock
    pyconrad = unittest.mock.MagicMock()

from os.path import dirname, join

import numpy as np
import skimage.io

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
    bm3d = Bm3d(lenna_noisy,
                   lenna_denoised,
                   BoxStencil(4, ndim),
                   BallStencil(2, ndim),
                   compilation_target='gpu',
                   max_block_matches=8,
                   threshold=0.1)


def main():
    test_bm3d()


if __name__ == '__main__':
    main()
