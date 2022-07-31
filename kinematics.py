import numpy as np
import globals_


def fk(q_start, sigma, v, sim_time):

    dt = 0.1
    t = np.arange(dt, sim_time + dt, dt)

    q_list = []
    q_list.append(q_start)

    for i in range(len(t)):
        q_current = q_list[i]
        J = hybridJacobian(q_start, q_current, sigma)

        q_dot = np.matmul(J, v)
        q_new = q_current + q_dot * dt

        q_list.append(q_new)

    return q_list


def hybridJacobian(q_start, q, sigma):

    sigma_rigid = int(not (sigma[0] or sigma[1]))
    J_rigid = sigma_rigid * np.array([[np.cos(q[2]), -np.sin(q[2]), 0],
                                      [np.sin(q[2]), np.cos(q[2]), 0],
                                      [0, 0, 1],
                                      [0, 0, 0],
                                      [0, 0, 0]])

    m = np.array([3 / 2, 1, 3 / 4])

    alpha = np.array(q[3:]) * globals_.L_VSS
    theta = np.divide(np.tile(np.reshape(alpha, (2, 1)), 3), m) - np.pi

    a = globals_.SPIRAL_COEF[0]
    b = np.tile(globals_.SPIRAL_COEF[1], (2, 1))
    for i in range(2):
        if q[i + 3] > 0:
            b[i] = -b[i]
            theta[i] += 2 * np.pi

    rho = a * np.exp(b * theta)

    Kappa = np.divide(m, globals_.L_VSS * rho)
    Phi = np.divide(m, rho)

    sigma_soft1 = int(not sigma[1] and sigma[0])
    sigma_soft2 = int(not sigma[0] and sigma[1])
    sigma_full_soft = int(sigma[0] and sigma[1])
    J_soft = np.array([[0, 0],
                       [0, 0],
                       [-sigma_soft2 * Phi[1, 1] - sigma_full_soft * Phi[1, 2],
                           sigma_soft1 * Phi[0, 1] + sigma_full_soft * Phi[0, 2]],
                       [sigma_soft1 * Kappa[0, 0] + sigma_full_soft * Kappa[0, 2],
                        sigma_soft1 * Kappa[0, 1] + sigma_full_soft * Kappa[0, 2]],
                       [sigma_soft2 * Kappa[1, 1] + sigma_full_soft * Kappa[1, 2], sigma_soft2 * Kappa[1, 0] + sigma_full_soft * Kappa[1, 2]]])

    J = np.concatenate((J_soft, J_rigid), axis=1)
    return J
    # theta1 = np.array([2 * q[3] * globals_.L_VSS / 3, q[3] *
    #                    globals_.L_VSS, 4 * q[3] * globals_.L_VSS / 3])
    # theta2 = np.array([2 * q[4] * globals_.L_VSS / 3, q[4] *
    #                    globals_.L_VSS, 4 * q[4] * globals_.L_VSS / 3])

    # if q[3] > 0:
    #     alpha1 = np.negative(globals_.SPIRAL_COEF[1]) * (theta1 + np.pi)
    # else:
    #     alpha1 = np.multiply(globals_.SPIRAL_COEF[1], theta1 - np.pi)

    # rho1 = np.multiply(globals_.SPIRAL_COEF[0], np.exp(alpha1))

    # if q[4] > 0:
    #     alpha2 = np.negative(globals_.SPIRAL_COEF[1]) * (theta2 + np.pi)
    # else:
    #     alpha2 = np.multiply(globals_.SPIRAL_COEF[1], theta2 - np.pi)

    # rho2 = np.multiply(globals_.SPIRAL_COEF[0], np.exp(alpha2))

    # # Jacobian matrix
    # a1 = q_start[2] + globals_.L_VSS * (q_start[4] - q[4])
    # b1 = q[4] * globals_.L_VSS / 3
    # c1 = q_start[2] + globals_.L_VSS * (q_start[4] - q[4] / 3)
    # d1 = q_start[2] + q[4] * globals_.L_VSS

    # a2 = q_start[2] - globals_.L_VSS * (q_start[3] - q[3])
    # b2 = q[3] * globals_.L_VSS / 3
    # c2 = q_start[2] - globals_.L_VSS * (q_start[3] - q[3] / 3)
    # d2 = q_start[2] - q[3] * globals_.L_VSS

    # j11 = [None] * 3
    # j12 = [None] * 3
    # j21 = [None] * 3
    # j22 = [None] * 3

    # j11[0] = -3 * globals_.L_LINK * np.sin(a1)
    # j11[1] = 8 * rho2[0] * np.sin(b1) * np.cos(c1)
    # j11[2] = -4 * rho2[0] * np.sin(d1)

    # j21[0] = 3 * globals_.L_LINK * np.cos(a1)
    # j21[1] = 8 * rho2[0] * np.sin(b1) * np.sin(c1)
    # j21[2] = 4 * rho2[0] * np.cos(d1)

    # j12[0] = -3 * globals_.L_LINK * np.sin(a2)
    # j12[1] = -8 * rho1[0] * np.sin(b2) * np.cos(c2)
    # j12[2] = -4 * rho1[0] * np.sin(d2)

    # j22[0] = 3 * globals_.L_LINK * np.cos(a2)
    # j22[1] = -8 * rho1[0] * np.sin(b2) * np.sin(c2)
    # j22[2] = 4 * rho1[0] * np.cos(d2)

    # pos1 = int(not sigma[0] and sigma[1]) / \
    #     (6 * rho2[1]) + int(sigma[0] and sigma[1]) / (8 * rho2[2])
    # pos2 = int(not sigma[1] and sigma[0]) / \
    #     (6 * rho1[1]) + int(sigma[0] and sigma[1]) / (8 * rho1[2])

    # J = np.array([[pos1 * np.sum(j11), pos2 * np.sum(j12), int(not (sigma[0] or sigma[1])) * np.cos(q[2]), int(not (sigma[0] or sigma[1])) * (-np.sin(q[2])), 0],
    #               [pos1 * np.sum(j21), pos2 * np.sum(j22), int(not (sigma[0] or sigma[1]))
    #                * np.sin(q[2]), int(not (sigma[0] or sigma[1])) * np.cos(q[2]), 0],
    #               [-(int(not sigma[0] and sigma[1]) / rho2[1] + int(sigma[0] and sigma[1]) * 3 / (4 * rho2[2])), int(not sigma[1]
    #                                                                                                                  and sigma[0]) / rho1[1] + int(sigma[0] and sigma[1]) * 3 / (4 * rho1[2]), 0, 0, int(not (sigma[0] or sigma[1]))],
    #               [int(not sigma[1] and sigma[0]) * 3 / (2 * globals_.L_VSS * rho1[0]) + int(sigma[0] and sigma[1]) * 3 / (4 * globals_.L_VSS * rho1[2]),
    #                int(not sigma[1] and sigma[0]) / (globals_.L_VSS * rho1[1]) + int(sigma[0] and sigma[1]) * 3 / (4 * globals_.L_VSS * rho1[2]), 0, 0, 0],
    #               [int(not sigma[0] and sigma[1]) / (globals_.L_VSS * rho2[1]) + int(sigma[0] and sigma[1]) * 3 / (4 * globals_.L_VSS * rho2[2]), int(not sigma[0] and sigma[1]) * 3 / (2 * globals_.L_VSS * rho2[0]) + int(sigma[0] and sigma[1]) * 3 / (4 * globals_.L_VSS * rho2[2]), 0, 0, 0]])

    # return J
