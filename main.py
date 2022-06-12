import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from math import *
from sympy import *
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from PIL import Image, ImageDraw
from initial_conditions import Y0
from cybership_odeint import cybership_ode

from functions import A, ld, B


if __name__ == '__main__':
    
    T_max = 350      # Time for which to simulate
    init_val = Y0    # initial values
    t = np.linspace(0,T_max,1000)

    sol = odeint(cybership_ode, init_val, t)

    s = sol[:,0]
    e = sol[:,1]
    z1 = sol[:,2]     
    z21 = sol[:,3]
    z22 = sol[:,4]
    z23 = sol[:,5]
    x = sol[:,9]
    y = sol[:,10]
    psi = sol[:,11]
    theta = sol[:,12]
    alpha1 = sol[:,13]
    alpha2 = sol[:,14]
    alpha3 = sol[:,15]

    xd = A*np.sin(theta/ld)*np.cos(theta/ld)
    yd = B*np.cos(theta/ld)


    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10,10))

    axs[0].set_title('Desired trajectory (blue) and actual trajectory (red)')
    line1, = axs[0].plot(y, x, color='r', label='actual trajectory')
    line2, = axs[0].plot(yd, xd, color='b', label='desired trajectory')
    axs[0].legend()

    axs[1].set_title('Cross-track error convergence to zero')
    axs[1].set_ylabel('cross-track error')
    axs[1].set_xlabel('time')
    line3, = axs[1].plot(t, e, color='g')
    plt.show()




