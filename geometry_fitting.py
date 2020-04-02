# ---------------------------------------------------------------------
# ---------------------------------------
# 
# python: 3.7
# date: 02/04/2020
# author: Xu Jing
# email: xj.yixing@hotmail.com
# 
# ---------------------------------------
# ----------------------------------------------------------------------

# ------------------- sphere ------------------------------------------------
'''
    R^2 = (x-x0)^2 + (y-y0)^2 + (z-z0)^2
'''
import numpy as np
from scipy.optimize import leastsq

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











