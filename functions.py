import numpy as np
from math import *
from sympy import *
from parameters import th


## Parametrization of trajectory

## Trajectory Parameters
A = 10                              # A
B = 15
ld = 15                             # Lambda


def tf_3(psi):
    mat = np.array([[cos(psi), -sin(psi), 0],
                    [sin(psi), cos(psi), 0],
                    [0, 0, 1]])
    return mat


def tf_2(psi):
    mat = np.array([[cos(psi), -sin(psi)],
                    [sin(psi), cos(psi)]])
    return mat


# Trajectory derivatives
# Returns symbolic expressions of the derivatives in term of th
def trajectory_derivatives():

    xd = A*sin(th/ld)*cos(th/ld)   
    yd = B*cos(th/ld)

    xd_d = diff(xd, th)     # (d/dth)(xd)
    yd_d = diff(yd, th)

    xd_dd = diff(xd_d, th)
    yd_dd = diff(yd_d, th)

    Xt = atan(yd_d/xd_d)
    Xt_d = diff(Xt, th)
    Xt_dd = diff(Xt_d, th)

    return xd, yd, xd_d, yd_d, xd_dd, yd_dd, Xt, Xt_d, Xt_dd