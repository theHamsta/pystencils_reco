# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import numpy as np

import pystencils
from pystencils_reco.resampling import downsample, scale_transform
from pystencils_reco.unet import max_pooling

try:
    import pyconrad.autoinit
except Exception:
    import unittest.mock
    pyconrad = unittest.mock.MagicMock()


def test_superresolution():

    x, y = np.random.rand(20, 10),  np.zeros((20, 10))

    kernel = scale_transform(x, y, 0.5).compile()
    print(pystencils.show_code(kernel))
    kernel()

    pyconrad.show_everything()


def test_downsample():
    shape = (20, 10)

    x, y = np.random.rand(*shape),  np.zeros(shape)

    kernel = downsample(x, y, 2).compile()
    print(pystencils.show_code(kernel))
    kernel()

    pyconrad.show_everything()
