import numpy as np
from math import *
from parameters import th, delta, gama, m, xg, Iz, h, K2, k1
from functions import tf_2, tf_3, trajectory_derivatives


# The Kinetics of Cybership II

def cybership_ode(Y,t):

    ## The states of the state-space model ##

    s = Y[0]                     # eplsilon includes [[s],[e]]
    e = Y[1]
    z1 = Y[2]                    # z1  = psi - psi_d
    z2_1 = Y[3]                  # z2 = nu - alpha
    z2_2 = Y[4]
    z2_3 = Y[5]
    bt_1 = Y[6]                  # bt = b_hat - b
    bt_2 = Y[7]
    bt_3 = Y[8]
    x = Y[9]                     # eta = [[x],[y],[psi]]   Global position and heading   
    y = Y[10]
    psi = Y[11]                
    theta = Y[12]                # theta   Path parametrization variable
    alpha_1 = Y[13]
    alpha_2 = Y[14]
    alpha_3 = Y[15]              # alpha = [[alpha_1],[alpha_2],[alpha_3]]

    # Variables in above state vector

    eps = np.array([[s], [e]])
    z2 = np.array([[z2_1],[z2_2],[z2_3]])
    bt = np.array([[bt_1],[bt_2],[bt_3]])
    eta = np.array([[x],[y],[psi]])
    alpha = np.array([[alpha_1],[alpha_2],[alpha_3]])

    ## vessel fixed velocities ##
    nu = z2 + alpha
    u_b = nu[0][0]
    v_b = nu[1][0]
    r_b = nu[2][0]

    # Speed #
    U = sqrt(u_b**2 + v_b**2)

    ## Earth fixed heading ##
    psi = eta[2][0]

    ## Rotation matrix R(psi) (SO(3)) ##
    R = tf_3(psi)

    ## Trajectory derivatives ##
    xds, yds, xd_ds, yd_ds, xd_dds, yd_dds, Xts, Xt_ds, Xt_dds = trajectory_derivatives()

    xd = xds.subs(th, theta)
    yd = yds.subs(th, theta)
    xd_d = xd_ds.subs(th, theta)
    yd_d = yd_ds.subs(th, theta)
    xd_dd = xd_dds.subs(th, theta)
    yd_dd = yd_dds.subs(th, theta)
    Xt = Xts.subs(th, theta)
    Xt_d = Xt_ds.subs(th, theta)
    Xt_dd = Xt_dds.subs(th, theta)
    
    # some required terms for eps_dot
    j = sqrt(xd_d**2 + yd_d**2)
    Xr = atan(-e/delta)
    U_pp = U*cos(Xr) + gama*s
    theta_dot = (U_pp)/j
    Xt_dot = Xt_d*theta_dot
    Rp = tf_2(Xr)
    Sp = np.array([[0, -Xt_dot],[Xt_dot, 0]])         # Skew symmetric matrix Sp

    eps_dot = np.transpose(Sp).dot(eps) + Rp.dot(np.array([[U],[0]])) - np.array([[U_pp],[0]])

    s_dot = eps_dot[0][0]
    e_dot = eps_dot[1][0]
    Xr_dot = (-delta/(delta**2 + e**2))*e_dot

    ## Hydrodynamic derivatives ##
    ## Surge hydrodynamic derivatives

    X_udot = -2.0
    X_u = -0.72253
    X_u_u = -1.32742
    X_uuu = -5.86643

    ## Sway hydrodynamic derivatives

    Y_vdot = -10.0
    Y_rdot = -0.0
    Y_v = -0.88965
    Y_r = -7.25
    Y_v_v = -36.47287
    Y_r_v = -0.805
    Y_v_r = -0.845
    Y_r_r = -3.45

    ## Yaw hydrodynamic derivatives

    N_vdot = -0.0
    N_rdot = -1.0
    N_v = 0.0313
    N_r = -1.9
    N_v_v = 3.95645
    N_r_v = 0.13
    N_v_r = 0.080
    N_r_r = -0.75


    ## Inertia matrix ##
    M_RB = np.array([[m, 0, 0],
                     [0, m, m*xg],
                     [0, m*xg, Iz]])
    M_A = np.array([[X_udot, 0, 0],
                    [0, -Y_vdot, -Y_rdot],
                    [0, -N_vdot, -N_rdot]])
    M = M_RB + M_A

    ## Centrifugal and coriolis matrix ##
    C_RB = np.array([[0, 0, -m*(xg*r_b + v_b)],
                     [0, 0, u_b],
                     [m*(xg*r_b + v_b), -m*u_b, 0]])
    C_A = np.array([[0, 0, Y_vdot*v_b+0.5*(N_vdot+Y_rdot)*r_b],
                    [0, 0, -X_udot*u_b],
                    [-Y_vdot*v_b-0.5*(N_vdot+Y_rdot)*r_b, X_udot*u_b, 0]])
    C = C_RB + C_A

    ## Hydrodynamic damping matrix ##
    D_L = np.array([[-X_u, 0, 0],
                    [0, -Y_v, -Y_r],
                    [0, -N_v, -N_r]])
    D_NL = np.array([[-X_u_u*abs(u_b)-X_uuu*(u_b**2), 0, 0],
                     [0, -Y_v_v*abs(v_b)-Y_r_v*abs(r_b), -Y_v_r*abs(v_b)-Y_r_r*abs(r_b)],
                     [0, -N_v_v*abs(v_b)-N_r_v*abs(r_b), -N_v_r*abs(v_b)-N_r_r*abs(r_b)]])
    D = D_L + D_NL
    
    psi_d = Xt + Xr
    psi_d_dot = Xt_dot + Xr_dot
    
    z2_dot = np.linalg.inv(M).dot(-C.dot(z2)-D.dot(z2)
                                    -np.transpose(R).dot(bt)
                                    -h.dot(z1)-K2.dot(z2))
    
    # Requirements for calculating psi_d_dotdot
    u_dot = z2_dot[0][0] 
    v_dot = z2_dot[1][0]
    U_dot = (u_b*u_dot + v_b*v_dot)/U
    theta_dotdot = ((j**2)*(U_dot*cos(Xr) 
                    - U*sin(Xr)*Xr_dot + gama*s_dot)
                    - U_pp*theta_dot*(xd_d*xd_dd + yd_d*yd_dd))/j**(3)
    Xt_dotdot = Xt_dd*(theta_dot**2) + Xt_d*theta_dotdot
    e_dotdot = -Xt_dotdot*s - Xt_dot*s_dot + U_dot*sin(Xr) + U*cos(Xr)*Xr_dot
    Xr_dotdot = ((delta**2 + e**2)*(-delta*e_dotdot) 
                + 2*e*delta*(e_dot**2))/((delta**2 + e**2)**2)
    psi_d_dotdot = Xt_dotdot + Xr_dotdot
    
    alpha_1_dot = 0
    alpha_2_dot = 0
    alpha_3_dot = (k1**2)*z1 - k1*z2[2][0] + psi_d_dotdot

    eps_dot = eps_dot            # Written above
    z1_dot = -k1*z1 + np.transpose(h).dot(z2)
    z2_dot = z2_dot              # Written above
    bt_dot = R.dot(z2)
    eta_dot = R.dot(nu)
    theta_dot = theta_dot        # Written above
    alpha_dot = np.array([[alpha_1_dot],[alpha_2_dot],[alpha_3_dot]])

    Yd = np.zeros((16,))
    Yd[0:2] = np.transpose(eps_dot)
    Yd[2] = z1_dot
    Yd[3:6] = np.transpose(z2_dot)
    Yd[6:9] = np.transpose(bt_dot)
    Yd[9:12] = np.transpose(eta_dot)
    Yd[12] = theta_dot
    Yd[13:16] = np.transpose(alpha_dot)

    return Yd