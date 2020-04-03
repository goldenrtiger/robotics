# ---------------------------------------------------------------------
# ---------------------------------------
# python: 3.7
# date: 02/04/2020
# author: Xu Jing
# email: xj.yixing@hotmail.com
# provide three methods to fit points to sphere.
# ---------------------------------------
# ----------------------------------------------------------------------

# ------------------- sphere ------------------------------------------------
'''
    R^2 = (x-x0)^2 + (y-y0)^2 + (z-z0)^2
'''
import numpy as np
from scipy.optimize import leastsq
import math

# test data: sphere centered at 'center' of radius 'R'
center = (np.random.rand(3) - 0.5) * 200
R = np.random.rand(1) * 100
coords = np.random.rand(100, 3) - 0.5
coords /= np.sqrt(np.sum(coords**2, axis=1))[:, None]
coords *= R
coords += np.random.normal(0, 0.5, [100,3]) # add some noise
coords += center

# inital 
p0 = [0, 0, 0, 1]

def fitfunc(p, coords):
    x0, y0, z0, R = p
    x, y, z = coords.T
    return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)

errfunc = lambda p, x: fitfunc(p, x) - p[3]

p1, flag = leastsq(errfunc, p0, args=(coords,))

print(f"center:{center}, R:{R}, p1:{p1}, flag:{flag}")

# -------------------------------- numpy lstsq -------------------------------------------
import numpy as np
#	fit a sphere to X,Y, and Z data points
#	returns the radius and center points of
#	the best fit sphere
def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX * spX) + (spY * spY) + (spZ * spZ)
    C, residules, rank, singval = np.linalg.lstsq(A, f, rcond=None)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2]

# -------------------------------- numpy lstsq -------------------------------------------
import tensorflow as tf
#	fit a sphere to X,Y, and Z data points
#	returns the radius and center points of
#	the best fit sphere
def tf_sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX * spX) + (spY * spY) + (spZ * spZ)
    C = tf.linalg.lstsq(A, f)
    # C = tf.linalg.lstsq(A, f, fast = False)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = tf.math.sqrt(t)

    return C[0], C[1], C[2], radius

x, y, z = coords.T
c0, c1, c2, radius = tf_sphereFit(x,y,z)

print(f"c0:{c0}, c1:{c1}, c2:{c2}, radius:{radius} ")









