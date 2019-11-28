# -*- coding: utf-8 -*-
from pkg_resources import DistributionNotFound, get_distribution

from pystencils_reco._assignment_collection import AssignmentCollection
from pystencils_reco._crazy_decorator import crazy, fixed_boundary_handling
from pystencils_reco._projective_matrix import ProjectiveMatrix
from pystencils_reco._typed_symbols import matrix_symbols, typed_symbols

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound


__all__ = ['AssignmentCollection',
           'crazy',
           'crazy_fixed_boundary_handling'
           'ProjectiveMatrix',
           'matrix_symbols',
           'typed_symbols',
           'ProjectiveMatrix',
           'fixed_boundary_handling',
           ]
