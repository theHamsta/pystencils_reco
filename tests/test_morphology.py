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

import pystencils
import pystencils_reco.morphology
from pystencils_reco.stencils import BallStencil

try:
    import pyconrad.autoinit
except ImportError:  # NOQA
    from unittest.mock import MagicMock
    pyconrad = MagicMock()


def test_morphology():
    import pytest
    pytest.importorskip("pycuda")
    from pycuda.gpuarray import to_gpu, zeros_like
    x = to_gpu((np.random.rand(21, 23, 25) > 0.9).astype(np.uint8))
    y = zeros_like(x)

    dilation = pystencils_reco.morphology.binary_dilation(x, y, BallStencil(1, ndim=3)).compile()

    for _ in range(1000000):
        dilation(x=x, y=y)
        x, y = y, x


def test_visualize_morphology():
    print(BallStencil(1, ndim=2))

    x, y = pystencils.fields('x,y: uint8[2d]')
    erosion = pystencils_reco.morphology.binary_erosion(x, y, BallStencil(1, ndim=2))
    print(erosion)
    erosion = erosion.compile()

    dilation = pystencils_reco.morphology.binary_dilation(x, y, BallStencil(1, ndim=2))
    print(dilation)
    dilation = dilation.compile()

    test_image = 1-skimage.io.imread(join(dirname(__file__), "test_data", "test_vessel2d_mask.png"), as_gray=True)
    test_image = np.ascontiguousarray(test_image, np.uint8)
    tmp = np.zeros_like(test_image)
    erosion(x=test_image, y=tmp)
    pyconrad.imshow(test_image, 'original')
    pyconrad.imshow(tmp, 'eroded')
    pyconrad.imshow(test_image-tmp, 'diff')

    dilation(x=test_image, y=tmp)
    pyconrad.imshow(test_image, 'original_dilation')
    pyconrad.imshow(tmp, 'dilated')
    pyconrad.imshow(test_image-tmp, 'diff_dilation')

    original = np.copy(test_image)
    erosion(x=test_image, y=tmp)
    dilation(x=tmp, y=test_image)

    pyconrad.imshow(original, 'original_opened')
    pyconrad.imshow(test_image, 'opened')
    pyconrad.imshow(test_image-original, 'diff_opening')

    # for i in tqdm.trange(20):
    # erosion(x=test_image, y=tmp)
    # tmp, test_image = test_image, tmp
    # pyconrad.imshow(test_image, str(i))
