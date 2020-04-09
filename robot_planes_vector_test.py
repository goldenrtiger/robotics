# ---------------------------------------------------------------------
# ---------------------------------------
# 
# python: 3.7
# date: 31/03/2020
# author: Xu Jing
# email: xj.yixing@hotmail.com
# 
# ---------------------------------------
# ----------------------------------------------------------------------
# ---------------------------------------------------------------------
# x + y + z = 0
# origin: (0,0,0), x-axis: (1,0,0), y-axis: (0,1,0), z-axis: (0,0,1)


import numpy as np
import math
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

# ------------ angle between two 3d vectors ----------------------
v1 = [1,0,0]
v2 = [0,-2,0]

def cal_angle_rad(vec1, vec2, acute):
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    if acute:
        return angle
    else:
        return 2 * np.pi - angle

print(cal_angle_rad(v1, v2, True))

# ------------- 3d points form a plane -----------------------------
# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
points = [[0.65612, 0.53440, 0.24175],
           [0.62279, 0.51946, 0.25744],
           [0.61216, 0.53959, 0.26394]]

p0, p1, p2 = points
x0, y0, z0 = p0
x1, y1, z1 = p1
x2, y2, z2 = p2

ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

point  = np.array(p0)
normal = np.array(u_cross_v)

d = -point.dot(normal)

xx, yy = np.meshgrid(range(10), range(10))

z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

# plot the surface
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, z)
plt.show()

# ----------------------- intersection between 3d Line and plane -----------------
# n: normal vector of the Plane 
# V0: any point that belongs to the Plane 
# P0: end point 1 of the segment P0P1
# P1:  end point 2 of the segment P0P1
n = np.array([1., 1., 1.])
V0 = np.array([1., 1., -5.])
P0 = np.array([-5., 1., -1.])
P1 = np.array([1., 2., 3.])

w = P0 - V0
u = P1 - P0
N = -np.dot(n,w)
D = np.dot(n,u)
sI = N / D
I = P0 + sI * u
print(I)

# -------------------- get z-axis from x-axis and y-axis -----------------------------------
'''
    (x axis) x (y axis) = (z axis)
    (y axis) x (z axis) = (x axis)
    (z axis) x (x axis) = (y axis)
'''
axis_x = [0.18525, -0.198557, -0.962423]
axis_y = [-0.791916, 0.549723, -0.265844]
origin = np.array([1061.244131, 95.25261, 752.979655])

ux, uy, uz = axis_x
vx, vy, vz = axis_y

u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx] # z axis
normal = np.array(u_cross_v)
d = -origin.dot(normal)

xx, yy = np.meshgrid(range(10), range(10))

z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

# plot the surface
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, z)
plt.show()

# --------------------- intersection of two planes ------------------------------
'''
    x + 2y + z - 1 = 0
    2x + 3y - 2z + 2 = 0

    {x = -7 + 7t, y = 4 - 4t}
    (x, y, z) = (-7 + 7t, 4 - 4t, t)
'''
x1, y1 = np.meshgrid(range(20), range(20))
t = np.linspace(0, 5)
x3 = -7 + 7 * t
y3 = 4 - 4 * t

# z1 = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
z1 = (-1 * x1 -2 * y1 - (-1))/1
z2 = (-2 * x1 -3 * y1 - (2))/(-2)
z3 = t

# plot the surface
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(x1, y1, z1, color='r')
plt3d.plot_surface(x1, y1, z2, color='b')
plt3d.plot(x3, y3, z3, color='tan')
plt.show()

# -------------------- get distance between a point and a line ---------------------------------
'''
    a line equation: ax + by + cz + d = 0
    a point: (x0, y0, z0)
    distance: abs(ax0 + by0 + cz0 + d)/sqrt(a**2 + b**2 + c**2)
'''
point = [0.,0.,0.]
line_normal = [1., 1., 1.]
a, b, c = line_normal
d = 10
x = np.linspace(-10, 10)
y = np.linspace(-10, 10)
z = (-d - a * x - b * y) / c
plt3d = plt.figure().gca(projection='3d')
plt3d.plot(x,y,z, color='b')
plt3d.scatter(point[0], point[1], point[2], c='r')

plt.show()

d = abs(a * point[0] + b * point[1] + c * point[2] + d)/math.sqrt(a ** 2 + b ** 2 + c ** 2)
print(f"distance:{d}")

# ------------------------ 2 3d points form a line --------------------
# => y - y1 = (y2 - y1)/(x2 - x1) * (x - x1)
# => z - z1 = (z2 - z1)/(x2 - x1) * (x - x1)
# or 
# (x - x1) / a = (y - y1) / b = (z - z1) / c
# a = (x2 - x1) / rou
# b = (y2 - y1) / rou
# c = (z2 - z1) / rou
# rou = srsqrtqt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
p0 = [0., 0., 0.]
p1 = [1., 1., 1.]

rou = math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2 )
a = (p1[0] - p0[0]) / rou
b = (p1[1] - p0[1]) / rou
c = (p1[2] - p0[2]) / rou

normal = np.array([a, b, c])
d = -np.array(p0).dot(normal)

x, y = np.linspace(0., 10.), np.linspace(0., 10.)
z = (-d - a * x - b * y) / c

plt3d = plt.figure().gca(projection='3d')
plt3d.plot(x,y,z, color='b')
plt3d.scatter(p0[0], p0[1], p0[2], c='r')

plt.show()




