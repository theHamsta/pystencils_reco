# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

from pystencils.field import FieldType
from pystencils_autodiff.field_tensor_conversion import ArrayWrapper
from pystencils_reco import crazy


@crazy
def foo(x, y):
    print(x)
    print(y)
    print("y.index_dimensions: " + str(y.index_dimensions))


def test_index_dimension_hack():
    import numpy as np

    x = np.zeros((20, 10))
    y = ArrayWrapper(np.zeros((20, 10)), 1, FieldType.CUSTOM)
    print(y.shape)

    foo(x, y)
