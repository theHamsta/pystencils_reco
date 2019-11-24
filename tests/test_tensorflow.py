# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pytest
import tensorflow as tf

import pystencils_reco.resampling
from pystencils_reco import crazy

pytest.importorskip('tensorflow')


def test_field_generation():

    x = tf.random.normal((20, 30, 100), name='x')
    y = tf.zeros((20, 30, 100), dtype=tf.float64)

    @crazy
    def kernel(x, y):
        print(x.dtype)
        print(y.dtype)
        return {x.center: y.center + 2}

    kernel(x, y)


@pytest.mark.parametrize('with_texture', (False,))
def test_texture(with_texture):

    x = tf.random.normal((20, 30, 100), name='x')
    y = tf.zeros_like(x, name='y')
    assignments = pystencils_reco.resampling.scale_transform(x, y, 2)

    kernel = assignments.compile()
    rtn = kernel()
    rtn = rtn[0].cpu()
    print(rtn)
    import pyconrad.autoinit
    pyconrad.show_everything()
