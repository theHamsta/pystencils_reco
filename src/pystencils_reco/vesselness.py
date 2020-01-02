import numpy as np
import sympy
from sympy import S

import pystencils
import pystencils_reco
from pystencils.data_types import TypedSymbol, create_type
from pystencils.math_optimizations import (
    ReplaceOptim, evaluate_constant_terms, optimize_assignments)
from pystencils.simp import sympy_cse
from pystencils_reco import crazy


class csqrt(sympy.Function):
    name = 'csqrt'
    nargs = 1

    def _eval_derivative(self, s):
        return sympy.sqrt(self.args[0]).diff(s)

    def fdiff(self, s):
        return sympy.sqrt(self.args[0]).diff(s)


use_complex_sqrt = ReplaceOptim(
    lambda e: e.is_Pow and e.args[1] == S('1/2'),
    lambda p: csqrt(p.args[0])
)


@crazy
def eigenvalues_3d(eig1, eig2, eig3, xx, xy, xz, yy, yz, zz):

    H = sympy.Matrix([[xx.center, xy.center, xz.center],
                      [xy.center, yy.center, yz.center],
                      [xz.center, yz.center, zz.center]]
                     )

    eigenvalues = list(H.eigenvals())

    assignments = pystencils.AssignmentCollection({
        [eig1.center, eig2.center, eig3.center][i]: sympy.re(eigenvalues[i]) for i in range(3)
    })

    class complex_symbol_generator():

        def __iter__(self):
            counter = 0
            while True:
                yield TypedSymbol('xi_%i' % counter, create_type(np.complex64))
                counter += 1

    assignments = pystencils.AssignmentCollection(optimize_assignments(assignments, [evaluate_constant_terms]))
    assignments.subexpression_symbol_generator = complex_symbol_generator()
    assignments = sympy_cse(assignments)
    assignments = optimize_assignments(assignments, [use_complex_sqrt])

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

    assignments = pystencils_reco.AssignmentCollection(assignments, perform_cse=False)
    return assignments


@crazy
def eigenvalues_3x3(eig1, eig2, eig3, xx, xy, xz, yy, yz, zz):
    """
    From https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices

    xx xy xz
    xy yy xy
    xz yz zz

    For real symetric eigenvalues we can use the following 'simple' algorithm:
    """
    from sympy.matrices import eye
    A = sympy.Matrix([[xx.center, xy.center, xz.center],
                      [xy.center, yy.center, yz.center],
                      [xz.center, yz.center, zz.center]])
    p1, p2, q, p, r, phi, e1, e2, e3 = sympy.symbols('p1, p2, q, p, r, phi, e1, e2, e3')

    B = (1 / p) * (A - q * eye(3))

    return {
        p1: xy.center ** 2 + xz.center ** 2 + yz.center ** 2,
        q: A.trace() / 3,
        p2: (xx.center - q) ** 2 + (yy.center - q) ** 2 + (zz.center - q) ** 2 + 2 * p1,
        p: sympy.sqrt(p2 / 6),
        r: B.det() * 0.5,
        phi: sympy.Piecewise(((sympy.pi / 3), (r <= -1)), ((0), (r >= 1)), ((sympy.acos(r) / 3), (True))),
        # phi: sympy.acos(r) / 3,
        e1: q + 2 * p * sympy.cos(phi),
        e3: q + 2 * p * sympy.cos(phi + (2 * sympy.pi/3)),
        e2: 3 * q - e1 - e3,
        eig1.center: e1,
        eig2.center: e2,
        eig3.center: e3,
    }
