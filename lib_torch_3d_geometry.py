import numpy as np
import math
import torch

class geometry_torch:
    @staticmethod
    def rad_between_two_lines(v1, v2, acute):
        '''
            theta = arccos(P.Q)/|P||Q| => |P| = 1, |Q| = 1
        '''
        rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        
        if acute:
            return rad
        else:
            return 2 * np.pi - rad

        return rad

    @staticmethod
    def get_vector_from_vector_rad_x(vector, rad):
        # vector = vector / np.linalg.norm(vector)
        # a, b = 1., 2.
        # # # a * v0[0] + b * v0[1] + c * v0[2] = cos(rad) * |P||Q|
        # value = np.cos(rad) - (a * vector[0] + b * vector[1])
        # c = np.divide(value, vector[2])
        # # c = 0 if np.isinf(c) or np.isneginf(c) else c
        # # v = np.array([a, b, c]) / np.linalg.norm([a, b, c])
        # # v = np.cos(rad) / vector
        # a , b = 1., 2.
        vector = vector / np.linalg.norm(vector)
        vector_yz = np.matrix([vector[1], vector[2]])
        R = np.matrix([[np.cos(rad), -np.sin(rad)],[np.sin(rad), np.cos(rad)]])
        v_yz = np.array(((R @ vector_yz.T).T))[0]
        v = np.append([vector[2]], [v_yz])
        
        return vector, v

    @staticmethod
    def create_plane_from_points(p0, p1, p2):
        p0, p1, p2 = np.array(p0), np.array(p1), np.array(p2)
        u = np.array([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]])
        v = np.array([p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]])
      
        normal = np.cross(u, v)
        a, b, c = normal.tolist()
        d = -p0.dot(normal)

        return np.array([a, b, c, d])

    @staticmethod
    def create_plane(origin, xaxis, yaxis, zaxis=np.zeros(3)):
        normal = torch.cross(xaxis, yaxis)
        a, b, c = normal[0], normal[1], normal[2]
        d = -origin.dot(normal)

        res = torch.zeros(4,1)
        res[0] = a
        res[1] = b
        res[2] = c
        res[3] = d

        return res.t()

    def create_orthogonal_planes_from_points(p0, p1, p2):
        '''
            ref: https://ww2.mathworks.cn/matlabcentral/answers/58244-how-to-create-2-orthogonal-planes-from-a-given-plane
        '''
        p0, p1, p2 = np.array(p0), np.array(p1), np.array(p2)
        N1 = (p1 - p0) 
        N2 = (p2 - p0) 

        N_Plane1 = np.cross(N1, N2)
        N_Plane2 = N1
        N_Plane3 = np.cross(N_Plane1, N_Plane2)

        return N_Plane1, N_Plane2, N_Plane3

    def create_orthogonal_planes_from_axis(xaxis, yaxis):
        N1, N2 = np.array(xaxis), np.array(yaxis)
        N_Plane1 = np.cross(N1, N2)
        N_Plane2 = N1
        N_Plane3 = np.cross(N_Plane1, N_Plane2)

        return N_Plane1, N_Plane2, N_Plane3
        

    @staticmethod
    def intersection_between_line_plane(plane, line):
        '''
            plane: np.array([a, b, c, d])
            line:[p0, v]
            ref: https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
        '''
        a, b, c, d = np.array(plane).tolist() ;   normal = np.array([a, b, c])
        x0, y0 = 1., 1. ;               z0 = (-d -a * x0 - b * y0) * 1. / c
        p = np.array([x0, y0, z0])

        p0 = line[0]
        v = line[1]
        p1 = p0 + v

        w = p0 - p
        u = p1 - p0

        N = -np.dot(normal, w)
        D = np.dot(normal, u)

        sI = N / D
        I = p0 + sI * u

        return I

    @staticmethod
    def intersection_between_2planes(origin_x, plane0, plane1):
        '''
            a1*x + b1*y + c1*z + d1 = 0
            a2*x + b2*y + c2*z + d2 = 0
            ref: http://mathcentral.uregina.ca/QQ/database/QQ.09.06/h/sarim1.html
        '''
        # a1, b1, c1, d1 = plane0[0], plane0[1], plane0[2], plane0[3]
        # a2, b2, c2, d2 = plane1[0], plane1[1], plane1[2], plane1[3]

        p, q, r = torch.cross(plane0[0,:3], plane1[0,:3])

        # x0 = torch.tensor([1.], dtype=torch.float32)
        x0 = origin_x
        param_bc = torch.zeros(2,2)
        param_bc[0, :] = plane0[0, 1:3]
        param_bc[1, :] = plane1[0, 1:3]
        value = torch.zeros(2,1)
        value[0] = -plane0[0,3]
        value[1] = -plane1[0,3]
        param_a = torch.zeros(2,1)
        param_a[0] = plane0[0, 0]
        param_a[1] = plane1[0, 0]

        value = value - param_a * x0
        
        p0, Lu = torch.solve(value, param_bc)
        point = torch.zeros(3,1)
        point[0] = x0
        point[1] = p0[0]
        point[2] = p0[1]

        v = torch.zeros(3, 1)
        v[0], v[1], v[2] = p, q, r

        return point.t(), v.t()

    @staticmethod
    def distance_between_point_line(point, line):
        '''
            point: M0
            line: M1, v
            ref: https://onlinemschool.com/math/library/analytic_geometry/p_line/
        '''
        M0, M1, v = point, line[0], line[1]
        
        return (torch.norm(torch.cross(M1 - M0, v)) / torch.norm(v))

    @staticmethod
    def intersection_between_two_lines(line0, line1):
        '''
            line equation: line1_p0 + t * v1

            P1 = (1,0,0)  V1 = (2,3,1)   line1: P1 + a * V1
            P2 = (0,5,5)  V2 = (5,1,-3)  line2: P2 + b * V2

            (P2-P1) x V2 = a(V1 x V2)  => get a. intersection point: P1 + a * V1

            ref: http://mathforum.org/library/drmath/view/63719.html
        '''
        p0, p1 = line0[0], line1[0]
        v0, v1 = line0[1], line1[1]

        cross0 = torch.cross(p1 - p0, v1)
        cross1 = torch.cross(v0, v1)

        t = torch.norm(cross0 / torch.norm(cross1, p=2))
        
        return p0 + t * v0
