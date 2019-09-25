# -*- coding: utf-8 -*-
#
# Distributed under terms of the GPLv3 license.

"""

"""

import sympy
import sympy.vector.orienters
from sympy import Matrix

import pystencils
import pystencils_reco
from pystencils.interpolation_astnodes import Interpolator

# def a_map(s_a1, s_a2, s_a3, p, q, r, s, t, u, t_z):
# return -p*s_a2 + q * s_a3 - q*t_z-r
##
# def b_map(s_a1, s_a2, s_a3, p, q, r, s, t, u, t_z):
# return p*s_a1 - s * s_a3 + s*t_z+t
##
# def c_map(s_a1, s_a2, s_a3, p, q, r, s, t, u, t_z):
# return -q*s_a1 + s * s_a2 - u;


def phi_u_map(kappa_1, kappa_2, kappa_3, rScale, radon_width):
    u = kappa_3 / sympy.sqrt((kappa_1 * kappa_1) + (kappa_2 * kappa_2))
    return u * rScale + (0.5 * (radon_width - 1.0))


def phi_v_map(kappa_1, kappa_2, offset_before_and_after, radon_height):
    return sympy.atan2(kappa_1, kappa_2) * radon_height / sympy.pi + 0.5 * radon_height + offset_before_and_after


def get_rotation_matrix(angle, axis):
    angle_rad = angle * sympy.pi / 180.0
    cos_theta = sympy.cos(angle_rad)
    sin_theta = sympy.sin(angle_rad)
    axis /= axis.norm()
    ux, uy, uz = axis
    u_skew = Matrix([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
    u_tensor = Matrix([[ux * ux, ux * uy, ux * uz], [ux * uy, uy * uy, uy * uz], [ux * uz, uy * uz, uz * uz]])
    id = sympy.eye(3)
    r = cos_theta * id + sin_theta * u_skew + (1 - cos_theta) * u_tensor
    R = sympy.eye(4)
    R[0, 0:3] = r[0, :]
    R[1, 0:3] = r[1, :]
    R[2, 0:3] = r[2, :]
    R[3, 3] = 1
    return R


sino_a, sino_b = pystencils.fields('sino_a, sino_b: float32[2d]')
theta = pystencils_reco.typed_symbols('theta', 'float32')


tex_sino_a = Interpolator(sino_a)
tex_sino_b = Interpolator(sino_b)

rScale, radon_width, offset_before_and_after, radon_height = sympy.symbols(
    "rScale, radon_width, offset_before_and_after, radon_heigth")

P_A, P_B, S_A, S_B = pystencils_reco.matrix_symbols("P_A, P_B, S_A, S_B", 'float32', 3, 4)

cam_a, cam_b = P_A, P_B

sp_a, sp_b = sympy.Matrix(pystencils_reco.typed_symbols('s_a:4', 'float32')
                          ), sympy.Matrix(pystencils_reco.typed_symbols('s_b:4', 'float32'))

baseline = Matrix(sp_a[:-1]) - Matrix(sp_b[:-1])
ortho_baseline = baseline.cross(sympy.Matrix([0, 0, 1]).T)
ortho_baseline

ortho_baseline_homo = Matrix([*ortho_baseline, 1])

rotation_matrix = get_rotation_matrix(theta, baseline)
print(rotation_matrix)

ep_point_dir = rotation_matrix @ ortho_baseline_homo

ep_point = sp_b + ep_point_dir


x1, y1, z1, w1 = ep_point
x2, y2, z2, w2 = sp_b
s_a1, s_a2, s_a3, w3 = sp_a

p = z1 - z2
q = y1 - y2
r = y1 * z2 - z1 * y2
s = x1 - x2
t = x1 * z2 - z1 * x2
u = x1 * y2 - y1 * x2

pluecker_line = Matrix([[0, p, -q, r], [-p, 0, s, -t], [q, -s, 0, u], [-r, t, -u, 0]])
epipolar_plane = sp_a.T @ pluecker_line
plane_infinity = sympy.Matrix([0, 0, 0, 1])


a1, b1, c1, d1 = epipolar_plane
a2, b2, c2, d2 = plane_infinity

u = -(c1 * d2 - c2 * d1)
t = b1 * d2 - b2 * d1
s = -(b1 * c2 - b2 * c1)
p = -(a1 * b2 - a2 * b1)
q = a1 * c2 - a2 * c1
r = -(a1 * d2 - a2 * d1)

ep_line_infinty = Matrix([[0, -u, -t, -s], [u, 0, -r, -q], [t, r, 0, -p], [s, q, p, 0]])

ep_line_a = cam_a @ ep_line_infinty @ cam_a.T
ep_line_b = cam_b @ ep_line_infinty @ cam_b.T

ep_line_a_x = -ep_line_a[0]
ep_line_a_y = ep_line_a[1]
ep_line_a_z = -ep_line_a[2]


ep_line_a_x = sympy.sign(ep_line_a_y) * ep_line_a_x
ep_line_a_y = sympy.sign(ep_line_a_y) * ep_line_a_y
ep_line_a_z = sympy.sign(ep_line_a_y) * ep_line_a_z


u_a = phi_u_map(ep_line_a_x, ep_line_a_y, ep_line_a_z, rScale, radon_width)
v_a = phi_v_map(ep_line_a_x, ep_line_a_y, offset_before_and_after, radon_height)


# line b
ep_line_b_x = -ep_line_b[0]
ep_line_b_y = ep_line_b[1]
ep_line_b_z = -ep_line_b[2]

ep_line_b_x = sympy.sign(ep_line_b_y) * ep_line_b_x
ep_line_b_y = sympy.sign(ep_line_b_y) * ep_line_b_y
ep_line_b_z = sympy.sign(ep_line_b_y) * ep_line_b_z


u_b = phi_u_map(ep_line_b_x, ep_line_b_y, ep_line_b_z, rScale, radon_width)
v_b = phi_v_map(ep_line_b_x, ep_line_b_y, offset_before_and_after, radon_height)


print(u_b)
print(v_b)

print(u_a)
print(v_a)
# interpolate on images...

# consval.spatial_shape[0]

consval = pystencils.fields('consval: [1d]')
d, tex_u_a, tex_v_a, tex_u_b, tex_v_b = pystencils_reco.typed_symbols('d, u_a, v_a, u_b, v_b', 'float32')

d_val = tex_sino_a.at(Matrix([tex_u_a, tex_v_a])) - tex_sino_b.at(Matrix([tex_u_b, tex_v_b]))
consistency_value = d * d / (1 + d**2)

assignments = pystencils_reco.AssignmentCollection({
    d: d_val,
    tex_u_a: u_a,
    tex_v_a: v_a,
    tex_u_b: u_b,
    tex_v_b: v_b,
    theta: sympy.pi * ((pystencils.x + 1) / (consval.spatial_shape[0] + 2)),
    consval.center(): consistency_value
}, perform_cse=True
)
print(assignments)
print(consistency_value)

kernel = assignments.compile('gpu')
print(kernel.code)

# Call kernel: kernel(consval=..., sino_a=..., ...)
# Array arguments must be pycuda.gpuarray.GpuArray!
