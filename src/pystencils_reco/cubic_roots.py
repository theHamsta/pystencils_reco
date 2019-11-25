import numpy as np
import sympy
import torch
from sympy.matrices.dense import hessian

import pystencils
import pystencils_autodiff._backport
from pystencils.simp import sympy_cse
from pystencils_reco import crazy


@crazy
def eigenvalues_3d(H_field: {'index_dimensions': 1}, xx, xy, xz, yy, yz, zz):
    # xx, xy, xz, yy, yz, zz = pystencils.fields('xx, xy, xz, yy, yz, zz: float32[3d]')

    H = sympy.Matrix([[xx.center, xy.center, xz.center],
                      [xy.center, yy.center, yz.center],
                      [xz.center, yz.center, zz.center]]
                     )

    eigenvalues = list(H.eigenvals())

    assignments = pystencils.AssignmentCollection({
        H_field.center(i): sympy.simplify(eigenvalues[i]) for i in range(3)
    })

    assignments = sympy_cse(assignments)
    assignments = pystencils.AssignmentCollection([pystencils.Assignment(a.lhs, sympy.re(a.rhs)) for a in assignments])
    assignments = sympy_cse(assignments)

    return assignments


@crazy
def vesselness(eigenvalues: {'index_dimensions': 1}, vesselness):

    lamda1 = sympy.max()


image = np.random.rand(30, 40, 50)
result = np.random.rand(30, 40, 50, 3)

#kernel = eigenvalues_3d(result, image, image, image, image, image, image).compile()

# kernel()


image = torch.randn((30, 40, 50))
result = torch.randn((30, 40, 50, 3))

kernel = eigenvalues_3d(result, image, image, image, image, image, image).compile()

kernel()
