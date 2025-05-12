import pybullet as p
import numpy as np
from math import *

from .rotation_utils import *


def isNearZero(a, tol=10e-6):
    if np.abs(a) < tol:
        return True
    else:
        return False


def TransInv(T):
    """
    Return the inverse matrix of transformation matrix T

    :param T: Transformation matrix
    :return: Inverse matrix of transformation matrix T
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3:4]

    Tinv = np.identity(4)
    Tinv[0:3, 0:3] = R.T
    Tinv[0:3, 3:4] = -R.T @ p
    return Tinv


def VecToso3(vec):
    """
    :param np.ndarray vec: [[wx],[wy],[wz]]

    :return: [[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]]
    :rtype: np.ndarray (3-by-3)
    """
    wx = vec[0, 0]
    wy = vec[1, 0]
    wz = vec[2, 0]

    return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])


def so3ToVec(so3):
    """
    :param np.ndarray so3: [[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]]

    :return: [[wx],[wy],[wz]]
    :rtype: np.ndarray (3-by-3)
    """
    wx = so3[2, 1]
    wy = so3[0, 2]
    wz = so3[1, 0]

    return np.array([[wx], [wy], [wz]])


def VecTose3(vec):
    """
    :param np.ndarray vec: [[vx], [vy], [vz], [wx], [wy], [wz]]

    :return: se3
    :rtype: np.ndarray (4-by-4)
    """
    v = vec[0:3, :]
    w = vec[3:6, :]
    w_ceil = VecToso3(w)
    top = np.concatenate([w_ceil, v], axis=1)
    btm = np.zeros([1, 4])

    return np.concatenate([top, btm], axis=0)


def se3ToVec(se3):
    w = so3ToVec(se3[0:3, 0:3])
    v = se3[0:3, [3]]

    return np.concatenate([v, w])


def Adjoint(T):
    """
    Return the Adjoint transformation matrix Adj of transformation matrix T

    :param T: Transformation matrix T
    :return: Adjoint transformation matrix Adj
    """
    R = T[0:3, 0:3]
    p = T[0:3, [3]]
    p_ceil = VecToso3(p)

    top = np.concatenate([R, p_ceil @ R], axis=1)
    btm = np.concatenate([np.zeros([3, 3]), R], axis=1)

    return np.concatenate([top, btm], axis=0)


def AdjointInv(T):
    """
    Return the Adjoint transformation matrix Adj of transformation matrix T

    :param T: Transformation matrix T
    :return: Adjoint transformation matrix Adj
    """
    R = T[0:3, 0:3]
    p = T[0:3, [3]]
    p_ceil = VecToso3(p)

    top = np.concatenate([R.T, -R.T @ p_ceil], axis=1)
    btm = np.concatenate([np.zeros([3, 3]), R.T], axis=1)

    return np.concatenate([top, btm], axis=0)


def adjoint(V):
    v = V[0:3, :]
    w = V[3:6, :]

    ad = np.zeros([6, 6])
    ad[0:3, 0:3] = VecToso3(w)
    ad[3:6, 3:6] = VecToso3(w)
    ad[0:3, 3:6] = VecToso3(v)

    return ad


def MatrixExp3(xi_ceil):
    return pin.exp3(so3ToVec(xi_ceil))


def MatrixExp6(lamb_ceil):
    return pin.exp6(se3ToVec(lamb_ceil)).np


def MatrixLog3(R):
    return VecToso3(pin.log3(R).reshape(-1, 1))


def MatrixLog6(T):
    return VecTose3(pin.log6(T).np.reshape(-1, 1))


def dexp3(xi):
    xi_ceil = VecToso3(xi)

    xi_norm = np.linalg.norm(xi)
    if isNearZero(xi_norm):
        return np.eye(3)
    else:
        alpha = sin(xi_norm) / xi_norm
        beta = 2 * (1 - np.cos(xi_norm)) / (xi_norm ** 2)

        return np.identity(3) + (beta / 2) * xi_ceil + ((1 - alpha) / (xi_norm ** 2)) * (xi_ceil @ xi_ceil)


def dexpInv3(xi):
    xi_ceil = VecToso3(xi)

    xi_norm = np.linalg.norm(xi)
    if isNearZero(xi_norm):
        return np.eye(3)
    else:
        alpha = sin(xi_norm) / xi_norm
        beta = 2 * (1 - np.cos(xi_norm)) / (xi_norm ** 2)
        gamma = alpha / beta

        return np.identity(3) - (1 / 2) * xi_ceil + ((1 - gamma) / (xi_norm ** 2)) * (xi_ceil @ xi_ceil)


def dexp6(lamb):
    eta = lamb[0:3, :]
    xi = lamb[3:6, :]

    dexp = dexp3(xi)

    res = np.zeros([6, 6])
    res[0:3, 0:3] = dexp
    res[3:6, 3:6] = dexp
    res[0:3, 3:6] = ddexp3(xi, eta)

    return res


def dexpInv6(lamb):
    eta = lamb[0:3, :]
    xi = lamb[3:6, :]

    dexpInv = dexpInv3(xi)

    res = np.zeros([6, 6])
    res[0:3, 0:3] = dexpInv
    res[3:6, 3:6] = dexpInv
    res[0:3, 3:6] = ddexpInv3(xi, eta)

    return res


def ddexp3(xi, dxi):
    xi_ceil = VecToso3(xi)
    dxi_ceil = VecToso3(dxi)

    xi_norm = np.linalg.norm(xi)
    if isNearZero(xi_norm):
        return 1 / 2 * dxi_ceil
    else:
        alpha = sin(xi_norm) / xi_norm
        beta = 2 * (1 - np.cos(xi_norm)) / (xi_norm ** 2)

        res = beta / 2 * dxi_ceil
        res += (1 - alpha) / xi_norm ** 2 * (xi_ceil @ dxi_ceil + dxi_ceil @ xi_ceil)
        res += (alpha - beta) / xi_norm ** 2 * np.dot(xi.T, dxi) * xi_ceil
        res += -1 / xi_norm ** 2 * (3 * (1 - alpha) / xi_norm ** 2 - beta / 2) * np.dot(xi.T, dxi) * xi_ceil @ xi_ceil

        return res


def ddexpInv3(xi, dxi):
    xi_ceil = VecToso3(xi)
    dxi_ceil = VecToso3(dxi)

    xi_norm = np.linalg.norm(xi)
    if isNearZero(xi_norm):
        return -1 / 2 * dxi_ceil
    else:
        alpha = sin(xi_norm) / xi_norm
        beta = 2 * (1 - np.cos(xi_norm)) / (xi_norm ** 2)
        gamma = alpha / beta

        res = -1 / 2 * dxi_ceil
        res += (1 - gamma) / xi_norm ** 2 * (xi_ceil @ dxi_ceil + dxi_ceil @ xi_ceil)
        res += 1 / xi_norm ** 2 * (1 / beta + gamma - 2) / xi_norm ** 2 * np.dot(xi.T, dxi) * xi_ceil @ xi_ceil

        return res


def ddexp6(lamb, dlamb):
    eta = lamb[0:3, :]
    deta = dlamb[0:3, :]
    xi = lamb[3:6, :]
    dxi = dlamb[3:6, :]

    ddexp = ddexp3(xi, dxi)

    res = np.zeros([6, 6])
    res[0:3, 0:3] = ddexp
    res[3:6, 3:6] = ddexp
    res[0:3, 3:6] = dddexp3(xi, dxi, eta, deta)

    return res


def ddexpInv6(lamb, dlamb):
    eta = lamb[0:3, :]
    deta = dlamb[0:3, :]
    xi = lamb[3:6, :]
    dxi = dlamb[3:6, :]

    ddexpInv = ddexpInv3(xi, dxi)

    res = np.zeros([6, 6])
    res[0:3, 0:3] = ddexpInv
    res[3:6, 3:6] = ddexpInv
    res[0:3, 3:6] = dddexpInv3(xi, dxi, eta, deta)
    return res


def dddexp3(xi, dxi, y=None, dy=None):
    if y is None:
        y = xi
    if dy is None:
        dy = dxi

    xi_ceil = VecToso3(xi)
    dxi_ceil = VecToso3(dxi)
    y_ceil = VecToso3(y)
    dy_ceil = VecToso3(dy)

    xi_norm = np.linalg.norm(xi)

    if isNearZero(xi_norm):
        return 1 / 2 * dy_ceil + 1 / 6 * (y_ceil @ dxi_ceil + dxi_ceil @ y_ceil)
    else:
        alpha = sin(xi_norm) / xi_norm
        beta = 2 * (1 - np.cos(xi_norm)) / (xi_norm ** 2)

        zeta = np.dot(xi.T, y) * np.dot(xi.T, dxi) / xi_norm ** 2
        G1 = (1 - alpha) / xi_norm ** 2
        G2 = (alpha - beta) / xi_norm ** 2
        G3 = (beta / 2 - 3 * G1) / xi_norm ** 2
        G4 = - G2 / beta
        G5 = (G1 + 2 * G2) / (beta * xi_norm ** 2)

        d0 = np.dot(dxi.T, y) + np.dot(xi.T, dy)
        d1 = np.dot(xi.T, y) * dxi_ceil + np.dot(xi.T, dxi) * y_ceil + (d0 - 4 * zeta) * xi_ceil
        d2 = np.dot(xi.T, y) * (xi_ceil @ dxi_ceil + dxi_ceil @ xi_ceil) + \
             np.dot(xi.T, dxi) * (xi_ceil @ y_ceil + y_ceil @ xi_ceil) + (d0 - 5 * zeta) * xi_ceil @ xi_ceil
        d3 = np.dot(xi.T, y) * (xi_ceil @ dxi_ceil + dxi_ceil @ xi_ceil) + \
             np.dot(xi.T, dxi) * (xi_ceil @ y_ceil + y_ceil @ xi_ceil) + (d0 - 3 * zeta) * xi_ceil @ xi_ceil

        res = beta / 2 * (dy_ceil - zeta * xi_ceil)
        res += G1 * (dy_ceil @ xi_ceil + xi_ceil @ dy_ceil + y_ceil @ dxi_ceil + dxi_ceil @ y_ceil + zeta * xi_ceil)
        res += G2 * (d1 + zeta * xi_ceil @ xi_ceil) + G3 * d2
        return res


def dddexpInv3(xi, dxi, y, dy):
    if y is None:
        y = xi
    if dy is None:
        dy = dxi

    xi_ceil = VecToso3(xi)
    dxi_ceil = VecToso3(dxi)
    y_ceil = VecToso3(y)
    dy_ceil = VecToso3(dy)

    xi_norm = np.linalg.norm(xi)

    if isNearZero(xi_norm):
        return -1 / 2 * dy_ceil + 1 / 12 * (y_ceil @ dxi_ceil + dxi_ceil @ y_ceil)
    else:
        alpha = sin(xi_norm) / xi_norm
        beta = 2 * (1 - np.cos(xi_norm)) / (xi_norm ** 2)
        gamma = alpha / beta

        zeta = np.dot(xi.T, y) * np.dot(xi.T, dxi) / xi_norm ** 2
        G1 = (1 - alpha) / xi_norm ** 2
        G2 = (alpha - beta) / xi_norm ** 2
        G3 = (beta / 2 - 3 * G1) / xi_norm ** 2
        G4 = - G2 / beta
        G5 = (G1 + 2 * G2) / (beta * xi_norm ** 2)

        d0 = np.dot(dxi.T, y) + np.dot(xi.T, dy)
        d1 = np.dot(xi.T, y) * dxi_ceil + np.dot(xi.T, dxi) * y_ceil + (d0 - 4 * zeta) * xi_ceil
        d2 = np.dot(xi.T, y) * (xi_ceil @ dxi_ceil + dxi_ceil @ xi_ceil) + \
             np.dot(xi.T, dxi) * (xi_ceil @ y_ceil + y_ceil @ xi_ceil) + (d0 - 5 * zeta) * xi_ceil @ xi_ceil
        d3 = np.dot(xi.T, y) * (xi_ceil @ dxi_ceil + dxi_ceil @ xi_ceil) + \
             np.dot(xi.T, dxi) * (xi_ceil @ y_ceil + y_ceil @ xi_ceil) + (d0 - 3 * zeta) * xi_ceil @ xi_ceil

        res = -1 / 2 * dy_ceil
        res += 2 / xi_norm ** 2 * (1 - gamma / beta) / xi_norm ** 2 * zeta * xi_ceil @ xi_ceil
        res += G4 * (dy_ceil @ xi_ceil + xi_ceil @ dy_ceil + y_ceil @ dxi_ceil + dxi_ceil @ y_ceil)
        res += G5 * d3

        return res