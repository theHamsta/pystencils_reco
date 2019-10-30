# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pystencils


def get_curl(input_field: pystencils.Field,
             curl_field: pystencils.Field):
    """Return a pystenicls.AssignmentCollection describing the calculation of
    the curl given a 2d or 3d vector field [z,y,x](f) or [y,x](f)

    Note that the curl of a 2d vector field is defined in ℝ3!
    Only the non-zero z-component is returned

    Arguments:
        field {pystenicls.Field} -- A field with index_dimensions <= 1
            Scalar fields are interpreted as a z-component

    Raises:
        NotImplementedError -- [description]
        NotImplementedError -- Only support 2d or 3d vector fields or scalar fields are supported

    Returns:
        pystenicls.AssignmentCollection -- AssignmentCollection describing the calculation of the curl
    """
    assert input_field.index_dimensions <= 1, "Must be a vector or a scalar field"
    assert curl_field.index_dimensions == 1, "Must be a vector field"
    discretize = pystencils.fd.Discretization2ndOrder(dx=1)

    if input_field.index_dimensions == 0:
        dy = pystencils.fd.Diff(input_field, 0)
        dx = pystencils.fd.Diff(input_field, 1)
        f_x = pystencils.Assignment(curl_field.center(0), discretize(dy))
        f_y = pystencils.Assignment(curl_field.center(1), discretize(dx))
        return pystencils.AssignmentCollection([f_x, f_y], [])

    else:

        if input_field.index_shape[0] == 2:
            raise NotImplementedError()

        elif input_field.index_shape[0] == 3:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
