import numpy as np
import sympy

import pystencils
import pystencils_reco
from pystencils.data_types import TypedSymbol, create_type
from pystencils.math_optimizations import evaluate_constant_terms, optimize_assignments
from pystencils.simp import sympy_cse
from pystencils_reco import crazy


@crazy
def eigenvalues_3d(eigenvaluefield: {'index_dimensions': 1}, xx, xy, xz, yy, yz, zz):

    H = sympy.Matrix([[xx.center, xy.center, xz.center],
                      [xy.center, yy.center, yz.center],
                      [xz.center, yz.center, zz.center]]
                     )

    eigenvalues = list(H.eigenvals())

    assignments = pystencils.AssignmentCollection({
        eigenvaluefield.center(i): sympy.re(eigenvalues[i]) for i in range(3)
    })

    class complex_symbol_generator():

        def __iter__(self):
            counter = 0
            while True:
                yield TypedSymbol('xi_%i' % counter, create_type(np.complex64))
                counter += 1

    assignments.subexpression_symbol_generator = complex_symbol_generator()

    assignments = sympy_cse(assignments)

    assignments = optimize_assignments(assignments, [evaluate_constant_terms])

    # complex_rhs = []
    # for a in assignments:
    # if isinstance(a.lhs, pystencils.Field.Access):
    # complex_rhs.append(a.rhs)
    # assignments = pystencils_reco.AssignmentCollection(assignments).subs(
    # {c: sympy.Function('real')(c) for c in complex_rhs})
    # print(assignments)

    # complex_symbols = [(a.lhs, a.rhs) for a in assignments if any(atom.is_real is False for atom in a.rhs.atoms())]

    # assignments = assignments.subs({a.lhs: a.rhs for a in assignments if any(
    # atom.is_real is False for atom in a.rhs.atoms())})
    # print(complex_symbols)

    return assignments


@crazy
def vesselness(eigenvalues: {'index_dimensions': 1}, vesselness):

    lamda1 = sympy.max()
