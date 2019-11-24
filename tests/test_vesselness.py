# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""


import sympy
import torch

import pystencils
from pystencils.backends.json import print_json, write_json
from pystencils_reco._assignment_collection import get_module_file
from pystencils_reco.vesselness import eigenvalues_3d


def test_vesselness():

    # import numpy as np

    # image = np.random.rand(30, 40, 50).astype(np.float32)
    # result = np.random.rand(30, 40, 50, 3).astype(np.float32)

    # assignments = eigenvalues_3d(result, image, image, image, image, image, image)
    # print(assignments)
    # print(assignments.reproducible_hash)
    # print(get_module_file(assignments, 'cuda'))
    # kernel = assignments.compile()
    # kernel()

    import torch
    torch.set_default_dtype(torch.double)

    result = torch.randn((2, 3, 4, 3))
    image0 = torch.zeros((2, 3, 4), requires_grad=True)
    image1 = torch.zeros((2, 3, 4), requires_grad=True)
    image2 = torch.zeros((2, 3, 4), requires_grad=True)
    image3 = torch.zeros((2, 3, 4), requires_grad=True)
    image4 = torch.zeros((2, 3, 4), requires_grad=True)
    image5 = torch.zeros((2, 3, 4), requires_grad=True)

    assignments = eigenvalues_3d(result, image0, image1, image2, image3, image4, image5)
    assignments.compile()
    # print(assignments)
    # # assignments.lambdify(sympy.
    # # kernel=assignments.compile()

    # main_assignments = [a for a in assignments if isinstance(a.lhs, pystencils.Field.Access)]
    # subexpressions = [a for a in assignments if not isinstance(a.lhs, pystencils.Field.Access)]
    # assignments = pystencils.AssignmentCollection(main_assignments, subexpressions)
    # lam = assignments.lambdify(sympy.symbols('xx_C xy_C xz_C yy_C yz_C zz_C'), module='tensorflow')

    # import tensorflow as tf

    # image0 = tf.random.uniform((20, 30, 40))
    # image1 = tf.random.uniform((20, 30, 40))
    # image2 = tf.random.uniform((20, 30, 40))
    # image3 = tf.random.uniform((20, 30, 40))
    # image4 = tf.random.uniform((20, 30, 40))
    # image5 = tf.random.uniform((20, 30, 40))

    # a = lam(image0, image1, image2, image3, image4, image5)
    import tensorflow as tf

    image0 = tf.random.uniform((20, 30, 40, 3, 3))
    eigenvalues, _ = tf.linalg.eigh(image0)

    print(eigenvalues)
    print(eigenvalues.shape)

    sorted_eigenvalues = tf.sort(eigenvalues, axis=-1)
    print(sorted_eigenvalues)

    l1 = sorted_eigenvalues[..., -1]
    l2 = sorted_eigenvalues[..., -2]
    l3 = sorted_eigenvalues[..., -3]
    print(l1)
    # assignments = assignments.new_without_subexpressions().main_assignments
    # lambdas = {assignment.lhs: sympy.lambdify(sympy.symbols('H_field xx xy xz yy yz zz'),
    # assignment.rhs, 'tensorflow') for assignment in assignments}
    # print(lambdas)

    # torch.autograd.gradcheck(kernel.apply, tuple(
    # [image0,
    # image1,
    # image2,
    # image3,
    # image4,
    # image5]),
    # atol=1e-4,
    # raise_exception=True)

    # image = tf.random.uniform((30, 40, 50))
    # result = tf.random.normal((30, 40, 50, 3))

    # #
    # kernel = eigenvalues_3d(result, image, image, image, image, image, image).compile()

    # kernel()
