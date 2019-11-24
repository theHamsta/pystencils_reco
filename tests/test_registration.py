# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pytest

from pystencils_reco.numpy_array_handler import NumpyArrayHandler
from pystencils_reco.registration import autocorrelation
from pystencils_reco.stencils import BallStencil


@pytest.mark.parametrize('array_handler', (NumpyArrayHandler(),))
def test_registration(array_handler):
    x = array_handler.randn((20, 30, 40))
    y = array_handler.randn((22, 32, 42))
    r_xx = array_handler.empty_like(y)
    r_yy = array_handler.empty_like(y)
    r_xy = array_handler.empty_like(y)

    assignments = autocorrelation(x, y, r_xx, r_yy, r_xy, BallStencil(3))

    print(assignments)

    kernel = assignments.compile()
    kernel()
