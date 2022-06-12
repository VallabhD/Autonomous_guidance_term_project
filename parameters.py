from sympy import *
import numpy as np

## Global variables and constant

th = Symbol('th')

## Ship model parameters
m = 23.8
Iz = 1.76
xg = 0.046
L = 1.255
ud = 0.25                           # Desired surge speed
b = np.array([[-1], [1], [0]])      # Environmental force disturbance 

## Guidance and Control parameters
delta = 3*L                         # Lookahead Distance
h = np.array([[0], [0], [1]])       # Projection Vector
k1 = 10                             # Constant > 0
k21 = 10 
k22 = 50 
k23 = 10                            # Control Gains
K2 = np.array([[k21, 0, 0],         # Control Gain Matrix
               [0, k22, 0],
               [0, 0, k23]])
gama = 100                         # Guidance Parameter