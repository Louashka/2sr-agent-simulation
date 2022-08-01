import numpy as np
import sympy as sym
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

    alpha = np.array(q[3:]) * globals_.L_VSS
    theta = np.divide(np.tile(np.reshape(alpha, (2, 1)), 3),
                      globals_.M) - np.pi

    a = globals_.SPIRAL_COEF[0]
    b = np.tile(globals_.SPIRAL_COEF[1], (2, 1))
    for i in range(2):
        if q[i + 3] > 0:
            b[i] = -b[i]
            theta[i] += 2 * np.pi

    rho = a * np.exp(b * theta)

    Kappa = np.divide(globals_.M, globals_.L_VSS * rho)
    Phi = np.divide(globals_.M, rho)

    sigma_soft1 = int(not sigma[1] and sigma[0])
    sigma_soft2 = int(not sigma[0] and sigma[1])
    sigma_full_soft = int(sigma[0] and sigma[1])

    Delta = np.zeros((2, 2))
    # Delta[:, 0] = (sigma_soft2 * Kappa[1, 1] + sigma_full_soft * Kappa[1, 2]) * np.array(pos[1].subs([(k, q[4]), (k0, q_start[4]),
    #                                                                                                   (phi, q_start[2]), (r, rho[1, 0])])).ravel()
    # Delta[:, 1] = (sigma_soft1 * Kappa[0, 1] + sigma_full_soft * Kappa[0, 2]) * np.array(pos[0].subs([(k, q[3]), (k0, q_start[3]),
    #                                                                                                   (phi, q_start[2]), (r, rho[0, 0])])).ravel()
    Delta[:, 0] = (sigma_soft2 * Kappa[1, 1] + sigma_full_soft * Kappa[1, 2]) * \
        getBodyFrame(rho[1, 0], q_start[2], q[4], q_start[4], 2).ravel()
    Delta[:, 1] = (sigma_soft1 * Kappa[0, 1] + sigma_full_soft * Kappa[0, 2]) * \
        getBodyFrame(rho[0, 0], q_start[2], q[3], q_start[3], 1).ravel()

    J_soft = np.array([[-sigma_soft2 * Phi[1, 1] - sigma_full_soft * Phi[1, 2],
                        sigma_soft1 * Phi[0, 1] + sigma_full_soft * Phi[0, 2]],
                       [sigma_soft1 * Kappa[0, 0] + sigma_full_soft * Kappa[0, 2],
                        sigma_soft1 * Kappa[0, 1] + sigma_full_soft * Kappa[0, 2]],
                       [sigma_soft2 * Kappa[1, 1] + sigma_full_soft * Kappa[1, 2],
                        sigma_soft2 * Kappa[1, 0] + sigma_full_soft * Kappa[1, 2]]])

    J_soft = np.concatenate((Delta, J_soft), axis=0)
    J = np.concatenate((J_soft, J_rigid), axis=1)
    return J


def getBodyFrame(r, phi, k, k0, seg):

    if seg == 1:
        pos = np.array([[-0.0266 * r * np.sin(phi + 0.0266 * k - 0.04 * k0) - 0.0006 * np.sin(phi + 0.04 * k - 0.04 * k0)],
                        [0.0266 * r * np.cos(phi + 0.0266 * k - 0.04 * k0) + 0.0006 * np.cos(phi + 0.04 * k - 0.04 * k0)]])
        # pos = bodyFramePosition(-1)
    elif seg == 2:
        pos = np.array([[-0.0266 * r * np.sin(phi - 0.0266 * k + 0.04 * k0) - 0.0006 * np.sin(phi - 0.04 * k + 0.04 * k0)],
                        [0.0266 * r * np.cos(phi - 0.0266 * k + 0.04 * k0) + 0.0006 * np.cos(phi - 0.04 * k + 0.04 * k0)]])
        # pos = bodyFramePosition(1)

    return pos


def bodyFramePosition(flag):

    x = sym.Symbol('x')
    y = sym.Symbol('y')
    r = sym.Symbol('r')
    phi = sym.Symbol(r'\phi')
    k = sym.Symbol('k')
    k0 = sym.Symbol('k\'')

    th = k * globals_.L_VSS / globals_.M[0]

    c_G = np.array([[-flag * r * sym.cos(th)],
                    [r * sym.sin(th)],
                    [1]])

    G_F = np.array([[-flag * globals_.L_LINK / 2],
                    [0],
                    [1]])

    cGth = - flag * k * globals_.L_VSS
    c_T_G = np.array([[sym.cos(cGth), -sym.sin(cGth), c_G[0].item()],
                      [sym.sin(cGth), sym.cos(cGth), c_G[1].item()],
                      [0, 0, 1]])

    c_F = sym.simplify(np.matmul(c_T_G, G_F))

    b_T_c = np.array([[1, 0, globals_.SPIRAL_CENTRE[0]],
                      [0, 1, globals_.SPIRAL_CENTRE[1]],
                      [0, 0, 1]])

    b_F = sym.simplify(np.matmul(b_T_c, c_F))

    o_T_b0 = np.array([[sym.cos(phi), -sym.sin(phi), x],
                       [sym.sin(phi), sym.cos(phi), y],
                       [0, 0, 1]])

    b0Bth = flag * k0 * globals_.L_VSS
    b0_T_b = np.array([[sym.cos(b0Bth), -sym.sin(b0Bth), flag * globals_.L_LINK / 2 + sym.sin(b0Bth) / k0],
                       [sym.sin(b0Bth), sym.cos(b0Bth),
                        (1 - sym.cos(b0Bth)) / k0],
                       [0, 0, 1]])

    o_T_b = sym.simplify(np.matmul(o_T_b0, b0_T_b))

    o_F = sym.simplify(np.matmul(o_T_b, b_F))

    diff = sym.Matrix(sym.simplify(sym.diff(o_F, k))[:2])

    return diff
