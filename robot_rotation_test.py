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
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fitdata = lambda arr: [[value[0][0] for value in arr]] + [[value[0][1] for value in arr]] + [[value[0][2] for value in arr]]

fRx = lambda t: np.matrix([[1, 0, 0],[0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
fRy = lambda t: np.matrix([[math.cos(t), 0, math.sin(t)],[0, 1, 0],  [-math.sin(t), 0, math.cos(t)]])
fRz = lambda t: np.matrix([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0,1]])

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
Rx = fRx(math.pi)
print(f"Rx: \n {Rx}, \n xaxis:\n {xaxis}")

origin_ = (Rx @ origin.T).T
xaxis_ = (Rx @ xaxis.T).T
yaxis_ = (Rx @ yaxis.T).T
zaxis_ = (Rx @ zaxis.T).T

arraxis_ = np.append([np.array(xaxis_), np.array(yaxis_)], [np.array(zaxis_)], axis = 0)
plotdata_ = np.append([np.array(origin), np.array(xaxis_), np.array(origin), np.array(yaxis_)], \
                [np.array(origin), np.array(zaxis_)], axis = 0)
data_ = fitdata(plotdata_)

M = np.linalg.solve(np.matrix(arraxis), np.matrix(arraxis_)).T

print(f"xaxis_:\n {xaxis_} \n yaxis_: \n {yaxis_} \n zaxis_:\n {zaxis_},\n M:\n {M} ")

ax.plot(data_[0], data_[1], data_[2], 'r')
plt.show()

