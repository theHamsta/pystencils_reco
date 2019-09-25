# -*- coding: utf-8 -*-
#
# Copyright © 2019 seitz_local <seitz_local@lmeXX>
#
# Distributed under terms of the GPLv3 license.

"""

"""
# import pycuda.gpuarray as gpuarray
import pystencils_reco
from pystencils import Field
from pystencils_reco.block_matching import block_matching_integer_offsets
from pystencils_reco.bm3d_kernels import (aggregate, apply_wieners,
                                          calc_wiener_coefficients,
                                          collect_patches, hard_thresholding)


class Bm3d:

    def __init__(self, input: Field,
                 output: Field,
                 block_stencil,
                 matching_stencil,
                 compilation_target,
                 max_block_matches,
                 blockmatching_threshold,
                 hard_threshold,
                 matching_function=pystencils_reco.functions.squared_difference,
                 wiener_sigma=None,
                 **compilation_kwargs):
        matching_stencil = sorted(matching_stencil, key=lambda o: sum(abs(o) for o in o))

        input_field = pystencils_reco._crazy_decorator.coerce_to_field('input_field', input)
        output_field = pystencils_reco._crazy_decorator.coerce_to_field('output_field', output)
        accumulated_weights = input_field.new_field_with_different_name('accumulated_weights')

        block_scores_shape = output_field.shape + (len(matching_stencil),)
        block_scores = Field.create_fixed_size('block_scores',
                                               block_scores_shape,
                                               index_dimensions=1,
                                               dtype=input_field.dtype.numpy_dtype)
        self.block_scores = block_scores

        block_matched_shape = input_field.shape + (max_block_matches, len(block_stencil))
        block_matched_field = Field.create_fixed_size('block_matched',
                                                      block_matched_shape,
                                                      index_dimensions=2,
                                                      dtype=input_field.dtype.numpy_dtype)
        self.block_matched_field = block_matched_field

        self.block_matching = block_matching_integer_offsets(input,
                                                             input,
                                                             block_scores,
                                                             block_stencil,
                                                             matching_stencil,
                                                             compilation_target,
                                                             matching_function,
                                                             **compilation_kwargs)
        self.collect_patches = collect_patches(block_scores,
                                               input,
                                               block_matched_field,
                                               block_stencil,
                                               matching_stencil,
                                               blockmatching_threshold,
                                               max_block_matches,
                                               compilation_target,
                                               **compilation_kwargs)
        complex_field = Field.create_fixed_size('complex_field',
                                                block_matched_shape + (2,),
                                                index_dimensions=3,
                                                dtype=input_field.dtype.numpy_dtype)
        group_weights = Field.create_fixed_size('group_weights',
                                                block_scores_shape,
                                                index_dimensions=1,
                                                dtype=input_field.dtype.numpy_dtype)
        self.complex_field = complex_field
        self.group_weights = group_weights
        self.hard_thresholding = hard_thresholding(
            complex_field, group_weights, hard_threshold).compile(compilation_target)
        if not wiener_sigma:
            wiener_sigma = pystencils_reco.typed_symbols('wiener_sigma', input_field.dtype.numpy_dtype)
        wiener_coefficients = Field.create_fixed_size('wiener_coefficients',
                                                      block_matched_shape,
                                                      index_dimensions=2,
                                                      dtype=input_field.dtype.numpy_dtype)
        self.get_wieners = calc_wiener_coefficients(complex_field, wiener_coefficients,
                                                    wiener_sigma).compile(compilation_target)
        self.apply_wieners = apply_wieners(complex_field, wiener_coefficients,
                                           group_weights).compile(compilation_target)

        self.aggregate = aggregate(block_scores,
                                   output,
                                   block_matched_field,
                                   block_stencil,
                                   matching_stencil,
                                   blockmatching_threshold,
                                   max_block_matches,
                                   compilation_target,
                                   group_weights,
                                   accumulated_weights,
                                   **compilation_kwargs)

    def do_on_gpu(self):
        pass
        # block_scores = gpuarray.zeros(self.block_scores.shape, self.block_scores.dtype.numpy_dtype)
        # weights = gpuarray.zeros(self.block_scores.shape, self.block_scores.dtype.numpy_dtype)
        # block_matched = gpuarray.zeros(self.block_matched_field.shape, self.block_matched_field.dtype.numpy_dtype)

        # print(self.block_matching.code)
        # print(self.aggregate.code)
        # print(self.collect_patches.code)

        # self.block_matching(block_scores=block_scores)
        # pyconrad.imshow(block_scores, 'block_scores')

        # self.collect_patches(block_scores=block_scores, block_matched=block_matched)
        # pyconrad.imshow(block_matched, 'block_matched')

        # print(block_matched.shape)
        # # fft[...] = block_matched

        # # reikna.fft.FFT(fft, (-3, -2, -1))
        # # pyconrad.imshow(fft, 'FFT')
        # import skcuda.fft as cufft
        # # forward_plan = cufft.cufftPlan3d(8, 4, 4, cufft.CUFFT_R2C)

        # plan = cufft.Plan((8, 4, 4), np.complex64, np.complex64, 512*512)
        # # forward_plan = cufft.cufftPlanMany(3, [8, 4, 4],
        # # 0, 0, 0,
        # # 0, 0, 0, cufft.CUFFT_R2C, 512*512)
        # # backward_plan = cufft.cufftPlanMany(3, [8, 4, 4],
        # # 0, 0, 0,
        # # 0, 0, 0, cufft.CUFFT_C2R, 512*512)

        # block_matched = block_matched.astype(np.complex64)
        # reshaped = pycuda.gpuarray.reshape(block_matched, (512*512, 8, 4, 4))
        # cufft.fft(reshaped, reshaped, plan)
        # pyconrad.imshow(reshaped, 'FFT')
        # print('fft')

        # real_shaped = pycuda.gpuarray.reshape(block_matched.view(np.float32), self.complex_field.shape)
        # self.hard_thresholding(complex_field=real_shaped, group_weights=weights)
        # self.wiener_filtering(complex_field=real_shaped, group_weights=weights)

        # cufft.ifft(reshaped, reshaped, plan, scale=True)
        # pyconrad.imshow(reshaped, 'iFFT')
        # print('ifft')

        # reshaped = pycuda.gpuarray.reshape(reshaped.real, (512, 512, 8, 16))

        # import pyconrad
        # self.aggregate(block_scores=block_scores, block_matched=reshaped)
        # pyconrad.imshow(lenna_denoised, 'denoised')
