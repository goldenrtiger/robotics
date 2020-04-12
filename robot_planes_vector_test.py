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
from lib_3d_geometry import geometry
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

p0 = np.array([1.,0.,0.])
p1 = np.array([0.,1.,0.])
p2 = np.array([0.,0.,1.])

plane = geometry.create_plane_from_points(p0, p1, p2)
a, b, c, d = plane.tolist()
x, y = np.meshgrid(range(10), range(10))
z = (-d - a * x  - b * y) * 1. / c

plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(x, y, z)
plt3d.scatter(p0[0],p0[1],p0[2], c='r')
plt3d.scatter(p1[0],p1[1],p1[2], c='b')
plt3d.scatter(p2[0],p2[1],p2[2], c='y')
plt3d.set_xlabel('$X$', fontsize=20)
plt3d.set_ylabel('$Y$', fontsize=20)
plt3d.set_zlabel('$Z$', fontsize=20, rotation = 0)
plt.show()

# ------------------------------------------------------------
xaxis, yaxis, zaxis = [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]
origin = [0.,0.,0.]
plane = geometry.create_plane(origin, xaxis, yaxis, zaxis)

a, b, c, d = plane.tolist()
x, y = np.meshgrid(range(10), range(10))
z = (-d - a * x  - b * y) * 1. / c

t = np.linspace(0., 10.)
x0 = origin[0] + t * xaxis[0]
y0 = origin[1] + t * xaxis[1]
z0 = origin[2] + t * xaxis[2]

x1 = origin[0] + t * yaxis[0]
y1 = origin[1] + t * yaxis[1]
z1 = origin[2] + t * yaxis[2]


plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(x, y, z)
plt3d.plot(x0,y0,z0, c='r')
plt3d.plot(x1,y1,z1, c='y')
plt3d.set_xlabel('$X$', fontsize=20)
plt3d.set_ylabel('$Y$', fontsize=20)
plt3d.set_zlabel('$Z$', fontsize=20, rotation = 0)

plt.show()

# -----------------------------------------------------------------------------
xaxis1, yaxis1 = [1.,0.,0.], [0.,1.,0.]
origin = [0.,0.,0.]

plane = geometry.create_plane(origin, xaxis1, yaxis1)
a, b, c, d = plane
x, y = np.meshgrid(range(20), range(20))
z = (-d - a * x  - b * y) * 1. / c

t = np.linspace(-2., 2.)
p0 = np.array([1., 1., 1.])
p1 = np.array([1., 2., 0.])
v0 = p1 - p0
x0 = p0[0] + t * v0[0]
y0 = p0[1] + t * v0[1]
z0 = p0[2] + t * v0[2]

point = geometry.intersection_between_line_plane(plane, [p0, v0])

plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(x, y, z, color='y')
plt3d.plot(x0,y0,z0, c='tan')
plt3d.scatter(point[0], point[1], point[2], c='r')
plt3d.set_xlabel('$X$', fontsize=20)
plt3d.set_ylabel('$Y$', fontsize=20)
plt3d.set_zlabel('$Z$', fontsize=20, rotation = 0)

plt.show()
# -----------------------------------------------------------------------------
xaxis1, yaxis1 = [1.,0.,0.], [0.,1.,0.]
xaxis2, yaxis2 = [2.,1.,1.], [3.,1.,1.]
origin = [0.,0.,0.]

plane1 = geometry.create_plane(origin, xaxis1, yaxis1)
plane2 = geometry.create_plane(origin, xaxis2, yaxis2)

p0, v0 = geometry.intersection_between_2planes(plane1, plane2)

a1, b1, c1, d1 = plane1.tolist()
xx1, yy1 = np.meshgrid(range(10), range(10))
zz1 = (-d1 - a1 * xx1  - b1 * yy1) * 1. / c1

a2, b2, c2, d2 = plane2.tolist()
xx2, yy2 = np.meshgrid(range(10), range(10))
zz2 = (-d2 - a2 * xx2  - b2 * yy2) * 1. / c2

t = np.linspace(-10., 10.)
x0 = p0[0] + t * v0[0]
y0 = p0[1] + t * v0[1]
z0 = p0[2] + t * v0[2]

plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx1, yy1, zz1, color='tan')
plt3d.plot_surface(xx2, yy2, zz2, color='y')
plt3d.plot(x0,y0,z0, c='r')
plt3d.set_xlabel('$X$', fontsize=20)
plt3d.set_ylabel('$Y$', fontsize=20)
plt3d.set_zlabel('$Z$', fontsize=20, rotation = 0)

plt.show()

# ----------------------------------------------------------------
p = np.array([0.,2.,3.])

p0 = np.array([3.,1.,-1.])
p1 = np.array([5.,2.,1.])

v = p1 - p0 
t = np.linspace(0.,10.)
x0 = p0[0] + t * v[0]
y0 = p0[1] + t * v[1]
z0 = p0[2] + t * v[2]

distance = geometry.distance_between_point_line(p, [p0, v])
print(f"distance:{distance}")

plt3d = plt.figure().gca(projection='3d')
plt3d.plot(x0,y0,z0, c='tan')
plt3d.scatter(p[0],p[1],p[2], c='r')

plt3d.set_xlabel('$X$', fontsize=20)
plt3d.set_ylabel('$Y$', fontsize=20)
plt3d.set_zlabel('$Z$', fontsize=20, rotation = 0)

plt.show()

# ----------------------------------------------------------------------
t = np.linspace(-10., 10.)

p0 = np.array([1., 0., 0.])
p1 = np.array([3., 3., 1.])
v0 = p1 - p0
x0 = p0[0] + t * v0[0]
y0 = p0[1] + t * v0[1]
z0 = p0[2] + t * v0[2]

p2 = np.array([0., 5., 5.])
p3 = np.array([5., 6., 2.])
v1 = p3 - p2
x1 = p2[0] + t * v1[0]
y1 = p2[1] + t * v1[1]
z1 = p2[2] + t * v1[2]

point = geometry.intersection_between_two_lines([p0, p1 - p0], [p2, p3 - p2])

plt3d = plt.figure().gca(projection='3d')
plt3d.plot(x0,y0,z0, c='y')
plt3d.plot(x1,y1,z1, c='tan')
plt3d.scatter(point[0], point[1], point[2], c='r')
plt3d.set_xlabel('$X$', fontsize=20)
plt3d.set_ylabel('$Y$', fontsize=20)
plt3d.set_zlabel('$Z$', fontsize=20, rotation = 0)

plt.show()

# -------------- two orthogonal planes ---------------------------------
# normal0 * normal1 = 0, normal0 * normal2 = 0
xaxis, yaxis, zaxis = [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]
origin = [0.,0.,0.]

plane0, plane1, plane2 = geometry.create_orthogonal_planes_from_axis(xaxis, yaxis)

a0, b0, c0 = plane0.tolist()
xx0, yy0 = np.meshgrid(range(10), range(10))
zz0 = (- a0 * xx0  - b0 * yy0) * 1. / c0

a1, b1, c1 = plane1.tolist()
xx1, yy1 = np.meshgrid(range(10), range(10))
zz1 = (- a1 * xx1  - b1 * yy1) * 1. / c1

a2, b2, c2 = plane2.tolist()
xx2, yy2 = np.meshgrid(range(10), range(10))
zz2 = (- a2 * xx2  - b2 * yy2) * 1. / c2

plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx0, yy0, zz0, color='tan')
plt3d.plot_surface(xx1, yy1, zz1, color='b')
plt3d.plot_surface(xx2, yy2, zz2, color='y')
plt3d.set_xlabel('$X$', fontsize=20)
plt3d.set_ylabel('$Y$', fontsize=20)
plt3d.set_zlabel('$Z$', fontsize=20, rotation = 0)
plt3d.set_xlim3d(-10, 10)
plt3d.set_ylim3d(-10, 10)
plt3d.set_zlim3d(-10, 10)

plt.show()

print(f"a0:{a0}, b0:{b0}, c0:{c0}")
print(f"a1:{a1}, b1:{b1}, c1:{c1}")
print(f"a2:{a2}, b2:{b2}, c2:{c2}")

# ---------------------- perpendicular vector -------------------------------
'''
    ref: https://math.stackexchange.com/questions/613232/find-equation-of-a-perpendicular-line-going-through-a-point
'''
p0 = np.array([1., 1., 1.])
p1 = np.array([1., 2., 0.])
v0 = p1 - p0

v0, v1 = geometry.get_vector_from_vector_rad_x(v0, 30 * np.pi / 180)

t = np.linspace(0., 10.)
x0 = p0[0] + t * v0[0]
y0 = p0[1] + t * v0[1]
z0 = p0[2] + t * v0[2]

x1 = p0[0] + t * v1[0]
y1 = p0[1] + t * v1[1]
z1 = p0[2] + t * v1[2]

plt3d = plt.figure().gca(projection='3d')
plt3d.plot(x0, y0, z0, color='tan')
plt3d.plot(x1, y1, z1, color='b')
plt3d.scatter(p0[0], p0[1], p0[2], color='y')
plt3d.scatter(p1[0], p1[1], p1[2], color='r')
plt3d.set_xlabel('$X$', fontsize=20)
plt3d.set_ylabel('$Y$', fontsize=20)
plt3d.set_zlabel('$Z$', fontsize=20, rotation = 0)
plt3d.set_xlim3d(-10, 10)
plt3d.set_ylim3d(-10, 10)
plt3d.set_zlim3d(-10, 10)

plt.show()

print(f"a1:{v0[0]}, b1:{v0[1]}, c1:{v0[2]}")
print(f"a2:{a}, b2:{b}, c2:{c}")
print(f"v0:{v0}, v1:{v1}, rad_between_two_lines:{geometry.rad_between_two_lines(v0, v1, True)}")



