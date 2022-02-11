import numpy as np
from math import *
from parameters import th, delta, k1, gama, ud, b
from functions import tf_2, trajectory_derivatives


# Initial State Values

eta0 = np.array([[0],[10],[1.57]])
nu0 = np.array([[0.25],[0],[0]])
bhat0 = np.array([[-1],[1],[0]])
theta0 = 0


######################################################################################

xds, yds, xd_ds, yd_ds, xd_dds, yd_dds, Xts, Xt_ds, Xt_dds = trajectory_derivatives()

xd0 = xds.subs(th, theta0)
yd0 = yds.subs(th, theta0)
xd0_d = xd_ds.subs(th, theta0)
yd0_d = yd_ds.subs(th, theta0)
xd0_dd = xd_dds.subs(th, theta0)
yd0_dd = yd_dds.subs(th, theta0)
Xt0 = Xts.subs(th, theta0)
Xt0_d = Xt_ds.subs(th, theta0)

# Inital positional error
Rp = tf_2(Xt0)
pd0 = np.array([[xd0],[yd0]])
p0 = eta0[0:2]
eps0 = np.transpose(Rp).dot(p0-pd0)
s0 = eps0[0][0]
e0 = eps0[1][0]

# Requirements for calculating psi_d0 and psi_d0_dot
j0 = sqrt(xd0_d**2 + yd0_d**2)
u_b0 = nu0[0][0]
v_b0 = nu0[1][0]
U0 = sqrt(u_b0**2 + v_b0**2)
Xr0 = atan(-e0/delta)
U_pp0 = U0*cos(Xr0) + gama*s0
theta0_dot = (U_pp0)/j0
Xt0_dot = Xt0_d*theta0_dot
e0_dot = -s0*Xt0_dot + U0*sin(Xr0)
Xr0_dot = (-delta/(delta**2 + e0**2))*e0_dot
psi_d0 = Xr0 + Xt0
psi_d0_dot = Xt0_dot + Xr0_dot

# Inital heading error
z1_0 = eta0[2][0] - psi_d0

# Initial velocity error
alpha0 = np.array([[ud], [0], [-k1*z1_0 + psi_d0_dot]])
z2_0 = nu0 - alpha0

bt0 = bhat0 - b

Y0 = np.zeros((16,))
Y0[0:2] = np.transpose(eps0)
Y0[2] = z1_0
Y0[3:6] = np.transpose(z2_0)
Y0[6:9] = np.transpose(bt0)
Y0[9:12] = np.transpose(eta0)
Y0[12] = 0
Y0[13:16] = np.transpose(alpha0)