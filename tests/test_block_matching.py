# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
# from os.path import dirname, join

# import numpy as np
# import pytest
# import skimage.io
# from tqdm import trange

# import pystencils
# import pystencils_reco
# from pystencils_reco.astnodes import ForEach
# from pystencils_reco.block_matching import (
    # block_matching_integer_offsets, block_matching_integer_offsets_unrolled,
    # single_block_matching)
# from pystencils_reco.stencils import BallStencil, BoxStencil

# try:
    # # import pyconrad.autoinit
    # raise ImportError('')
# except ImportError:  # NOQA
    # from unittest.mock import MagicMock
    # pyconrad = MagicMock()


# def test_block_matching_unrolled():

    # block_stencil = BallStencil(2, ndim=2)
    # matching_stencil = BallStencil(2, ndim=2)

    # x, y, matches = pystencils.fields('x,y, matches(%i): float32[2d]' % len(matching_stencil))
    # block_matching = block_matching_integer_offsets_unrolled(x, y, matches, block_stencil, matching_stencil)
    # print(block_matching)
    # block_matching = block_matching.compile('cpu')

    # test_image = 1 - skimage.io.imread(join(dirname(__file__), "test_data", "test_vessel2d_mask.png"), as_gray=True)
    # test_image = np.ascontiguousarray(test_image, np.float32)
    # result = np.zeros([*test_image.shape, len(matching_stencil)], np.float32)
    # block_matching(x=test_image, y=test_image, matches=result)
    # pyconrad.imshow(test_image, 'original')
    # pyconrad.imshow(test_image, 'compare')
    # pyconrad.imshow(np.swapaxes(result, 0, -1), 'result')


# def test_block_matching():

    # block_stencil = BallStencil(3, ndim=2)
    # matching_stencil = BallStencil(3, ndim=2)

    # x, y, matches = pystencils.fields('x,y, matches(%i): float32[2d]' % len(matching_stencil))
    # block_matching = block_matching_integer_offsets(
        # x, y, matches, block_stencil, matching_stencil, compilation_target='cpu')
    # print(block_matching.code)

    # test_image = 1 - skimage.io.imread(join(dirname(__file__), "test_data", "test_vessel2d_mask.png"), as_gray=True)
    # test_image = np.ascontiguousarray(test_image, np.float32)
    # result = np.zeros([*test_image.shape, len(matching_stencil)], np.float32)
    # block_matching(x=test_image, y=test_image, matches=result)
    # pyconrad.imshow(test_image, 'original')
    # pyconrad.imshow(test_image, 'compare')
    # pyconrad.imshow(np.swapaxes(result, 0, -1), 'result')


# def test_block_matching_gpu():
    # from pycuda.gpuarray import to_gpu

    # block_stencil = BallStencil(3, ndim=2)
    # matching_stencil = BallStencil(3, ndim=2)

    # x, y, matches = pystencils.fields('x,y, matches(%i): float32[2d]' % len(matching_stencil))
    # block_matching = block_matching_integer_offsets(
        # x, y, matches, block_stencil, matching_stencil, compilation_target='gpu')
    # print(block_matching.code)

    # test_image = 1 - skimage.io.imread(join(dirname(__file__), "test_data", "test_vessel2d_mask.png"), as_gray=True)
    # test_image = np.ascontiguousarray(test_image, np.float32)
    # result = np.zeros([*test_image.shape, len(matching_stencil)], np.float32)
    # test_image = to_gpu(test_image)
    # result = to_gpu(result)
    # block_matching(x=test_image, y=test_image, matches=result)
    # pyconrad.imshow(test_image, 'original')
    # pyconrad.imshow(test_image, 'compare')
    # pyconrad.imshow(np.swapaxes(result.get(), 0, -1), 'result')


# @pytest.mark.skip("long failing test")
# def test_larger_blocks():
    # block_stencil = BoxStencil(3, ndim=2)
    # matching_stencil = BallStencil(3, ndim=2)

    # print('forward')
    # x, y, matches = pystencils.fields('x,y, matches(%i): float32[2d]' % len(matching_stencil))
    # block_matching = block_matching_integer_offsets_unrolled(x, y, matches, block_stencil, matching_stencil)
    # print(str(len(block_matching.all_assignments)) + " assignments")
    # print('compile')
    # _ = block_matching.compile('cpu')
    # print('backward')
    # backward = block_matching.backward()
    # print('print')
    # print(backward.compile())
    # # block_matching.backward().compile()
    # print('backward')


# def test_combination_single_block_matching():
    # block_stencil = BoxStencil(9, ndim=2)
    # matching_stencil = BallStencil(5, ndim=2)

    # x, y, matches = pystencils.fields('x,y, matches(%i): float32[2d]' % len(matching_stencil))
    # offset = pystencils_reco.typed_symbols('o:2', 'int32')
    # i = pystencils_reco.typed_symbols('i', 'int32')
    # print(len(matching_stencil))
    # block_matching = single_block_matching(x, y, matches, block_stencil, offset, i)
    # print('backward')
    # backward = block_matching.backward().compile('gpu', ghost_layers=0)
    # print('print')
    # print(backward.code)


# def test_for_each():
    # block_stencil = BoxStencil(9, ndim=2)
    # matching_stencil = BallStencil(3, ndim=2)

    # x, y, matches = pystencils.fields('x,y, matches(%i): float32[2d]' % len(matching_stencil))
    # offset = pystencils_reco.typed_symbols('o:2', 'int32')
    # i = pystencils_reco.typed_symbols('i', 'int32')
    # block_matching = single_block_matching(x, y, matches, block_stencil, offset, i)
    # ast = pystencils.create_kernel(block_matching, target='gpu', data_type='float', ghost_layers=0)
    # ast._body = ForEach(ast.body, offset, matching_stencil, i)
    # # ast = ForLoop(ast.body,  i, 0, 10, 2)
    # print(pystencils.show_code(ast))
    # kernel = pystencils.gpucuda.cudajit.make_python_function(ast)


# def test_for_each_3d():
    # block_stencil = BoxStencil(2, ndim=3)
    # matching_stencil = BallStencil(1, ndim=3)

    # x, y, matches = pystencils.fields('x,y, matches(%i): float32[3d]' % len(matching_stencil))
    # offset = pystencils_reco.typed_symbols('o:3', 'int32')
    # i = pystencils_reco.typed_symbols('i', 'int32')
    # block_matching = single_block_matching(x, y, matches, block_stencil, offset, i)
    # ast = pystencils.create_kernel(block_matching, target='gpu', data_type='float', ghost_layers=0)
    # ast._body = ForEach(ast.body, offset, matching_stencil, i)
    # # ast = ForLoop(ast.body,  i, 0, 10, 2)
    # print(pystencils.show_code(ast))
    # kernel = pystencils.gpucuda.cudajit.make_python_function(ast)


# def test_block_matching_unrolled_gpu():
    # import pycuda.autoinit  # NOQA
    # from pycuda.gpuarray import to_gpu, zeros

    # block_stencil = BallStencil(2, ndim=2)
    # matching_stencil = BallStencil(2, ndim=2)

    # x, y, matches = pystencils.fields('x,y, matches(%i): float32[2d]' % len(matching_stencil))
    # block_matching = block_matching_integer_offsets_unrolled(x, y, matches, block_stencil, matching_stencil)
    # # print(block_matching)
    # block_matching = block_matching.compile('gpu')

    # test_image = 1 - skimage.io.imread(join(dirname(__file__), "test_data", "test_vessel2d_mask.png"), as_gray=True)
    # test_image = to_gpu(np.ascontiguousarray(test_image, np.float32))
    # result = zeros([*test_image.shape, len(matching_stencil)], np.float32)
    # for i in trange(1000):
        # block_matching(x=test_image, y=test_image, matches=result)
    # pyconrad.imshow(test_image, 'original')
    # pyconrad.imshow(test_image, 'compare')
    # pyconrad.imshow(np.swapaxes(result.get(), 0, -1), 'result')
