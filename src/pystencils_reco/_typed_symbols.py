# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""
Helper functions for creating explicitly typed symbols.
"""

from collections.abc import Iterable

import sympy as sp

from pystencils.data_types import TypedSymbol


def typed_symbols(names, dtype, *args):
    symbols = sp.symbols(names, *args)
    if isinstance(symbols, Iterable):
        return tuple(TypedSymbol(str(s), dtype) for s in symbols)
    else:
        return TypedSymbol(str(symbols), dtype)


def matrix_symbols(names, dtype, rows, cols):
    if isinstance(names, str):
        names = names.replace(' ', '').split(',')

    matrices = []
    for n in names:
        symbols = typed_symbols("%s:%i" % (n, rows * cols), dtype)
        matrices.append(sp.Matrix(rows, cols, lambda i, j: symbols[i * cols + j]))

    if len(matrices) == 1:
        return matrices[0]
    else:
        return tuple(matrices)
