# ---------------------------------------------------------------------
# ---------------------------------------
# 
# python: 3.7
# date: 31/03/2020
# author: Xu Jing
# email: xj.yixing@hotmail.com
# 
# 1. get rotation matrix from EulerAngles
# 2. get EulerAngles from rotation matrix
# ---------------------------------------
# ----------------------------------------------------------------------
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
  
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) : 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


fitdata = lambda arr: [[value[0][0] for value in arr]] + [[value[0][1] for value in arr]] + [[value[0][2] for value in arr]]

fRx = lambda t: np.matrix([[1, 0, 0],[0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
fRy = lambda t: np.matrix([[math.cos(t), 0, math.sin(t)],[0, 1, 0],  [-math.sin(t), 0, math.cos(t)]])
fRz = lambda t: np.matrix([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0,1]])

# ---------------------- rotate around a vector ----------------------------------------
def get_rotation(rxyz_rad, vector, origin = np.array([0., 0., 0.])):
    ox, oy, oz = origin.tolist()
    ux_, uy_, uz_ = vector.tolist()
    ux = ux_ - ox; uy = uy_ - oy; uz = uz_ - oz

    R = [
            [
                math.cos(rxyz_rad) + ux ** 2 * (1 - math.cos(rxyz_rad)), 
                ux * uy * (1 - math.cos(rxyz_rad)) - uz * math.sin(rxyz_rad),
                ux * uz * (1 - math.cos(rxyz_rad)) + uy * math.sin(rxyz_rad)
            ], 
            [
                uy * ux * (1 - math.cos(rxyz_rad)) + uz * math.sin(rxyz_rad), 
                math.cos(rxyz_rad) + uy ** 2 * (1 - math.cos(rxyz_rad)),
                uy * uz * (1 - math.cos(rxyz_rad)) - ux * math.sin(rxyz_rad)
            ], 
            [
                uz * ux * (1 - math.cos(rxyz_rad)) - uy * math.sin(rxyz_rad), 
                uz * uy * (1 - math.cos(rxyz_rad)) + ux * math.sin(rxyz_rad),
                math.cos(rxyz_rad) + uz ** 2 * (1 - math.cos(rxyz_rad))
            ], 
        ]

    return np.matrix(R)

def normalize_vector(vector):
    return vector / np.sqrt(np.sum(vector ** 2))

origin = np.matrix([0, 0, 0])
xaxis = np.matrix([1, 0, 0])
yaxis = np.matrix([0, 1, 0])
zaxis = np.matrix([0, 0, 1])

arraxis = np.append([np.array(xaxis),  np.array(yaxis)], [np.array(zaxis)], axis = 0)
plotdata = np.append([np.array(origin), np.array(xaxis), np.array(origin), np.array(yaxis)], \
                [np.array(origin), np.array(zaxis)], axis = 0)
data = fitdata(plotdata)

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(data[0], data[1], data[2], 'b')

# -------------------- rotate aixses about the x-axis by xx degrees -----------
Rx = fRx(math.pi/2)
Ry = fRy(math.pi/4)
print(f"Rx: \n {Rx}, \n xaxis:\n {xaxis}")

origin_ = (Rx @ origin.T).T
xaxis_ = (Rx @ xaxis.T).T
yaxis_ = (Rx @ yaxis.T).T
zaxis_ = (Rx @ zaxis.T).T

origin_ = (Ry @ origin_.T).T
xaxis_ = (Ry @ xaxis_.T).T
yaxis_ = (Ry @ yaxis_.T).T
zaxis_ = (Ry @ zaxis_.T).T

# ---------------------- rotate around a vector ----------------------------------------
vector = normalize_vector(np.array([-1.438783, 13.919369, -99.016065]))
Rx = get_rotation(0.5035549257762054, vector)
xaxis_ = (Rx @ np.matrix(vector).T).T

arraxis_ = np.append([np.array(xaxis_), np.array(yaxis_)], [np.array(zaxis_)], axis = 0)
plotdata_ = np.append([np.array(origin), np.array(xaxis_), np.array(origin), np.array(yaxis_)], \
                [np.array(origin), np.array(zaxis_)], axis = 0)
data_ = fitdata(plotdata_)

# M = np.linalg.solve(np.matrix(arraxis), np.matrix(arraxis_))
R = np.matrix(arraxis) @ np.linalg.inv(np.matrix(arraxis_))
arraxis1 = R @ np.matrix(arraxis_)

print(f"xaxis_:\n {xaxis_} \n yaxis_: \n {yaxis_} \n zaxis_:\n {zaxis_},\n M:\n {M} ")

ax.plot(data_[0], data_[1], data_[2], 'r')
plt.show()

# -------------------- get EulerAngles from rotation matrix -----------------
rotationMatrixToEulerAngles(M)
