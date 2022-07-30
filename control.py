import numpy as np
from queue import Queue
import graphics


class Config(object):
    def __init__(self, q, t):
        self.q = q
        self.t = t


class Control:

    # Constant parameters
    l_0 = 30 * 10**(-3)  # plastic link length
    l = 40 * 10**(-3)  # VSS length
    log_spiral_coef = np.array(
        [[2.3250 * l, 0.3165], [3.3041 * l, 0.083], [2.4471 * l, 0.2229]])

    dt = 0.1  # step size

    def __init__(self, q_0):
        self.q_0 = q_0  # Initial configuration

    def hybridJacobian(self, q, s1, s2):
        theta1 = np.array([2 * q[3] * self.l / 3, q[3] *
                           self.l, 4 * q[3] * self.l / 3])
        theta2 = np.array([2 * q[4] * self.l / 3, q[4] *
                           self.l, 4 * q[4] * self.l / 3])

        if q[3] > 0:
            alpha1 = np.multiply(-self.log_spiral_coef[:, 1], theta1 + np.pi)
        else:
            alpha1 = np.multiply(self.log_spiral_coef[:, 1], theta1 - np.pi)

        rho1 = np.multiply(self.log_spiral_coef[:, 0], np.exp(alpha1))

        if q[4] > 0:
            alpha2 = np.multiply(-self.log_spiral_coef[:, 1], theta2 + np.pi)
        else:
            alpha2 = np.multiply(self.log_spiral_coef[:, 1], theta2 - np.pi)

        rho2 = np.multiply(self.log_spiral_coef[:, 0], np.exp(alpha2))

        # Jacobian matrix
        a1 = self.q_0[2] + self.l * (self.q_0[4] - q[4])
        b1 = q[4] * self.l / 3
        c1 = self.q_0[2] + self.l * (self.q_0[4] - q[4] / 3)
        d1 = self.q_0[2] + q[4] * self.l

        a2 = self.q_0[2] - self.l * (self.q_0[3] - q[3])
        b2 = q[3] * self.l / 3
        c2 = self.q_0[2] - self.l * (self.q_0[3] - q[3] / 3)
        d2 = self.q_0[2] - q[3] * self.l

        j11 = [None] * 3
        j12 = [None] * 3
        j21 = [None] * 3
        j22 = [None] * 3

        j11[0] = -3 * self.l_0 * np.sin(a1)
        j11[1] = 8 * rho2[0] * np.sin(b1) * np.cos(c1)
        j11[2] = -4 * rho2[0] * np.sin(d1)

        j21[0] = 3 * self.l_0 * np.cos(a1)
        j21[1] = 8 * rho2[0] * np.sin(b1) * np.sin(c1)
        j21[2] = 4 * rho2[0] * np.cos(d1)

        j12[0] = -3 * self.l_0 * np.sin(a2)
        j12[1] = -8 * rho1[0] * np.sin(b2) * np.cos(c2)
        j12[2] = -4 * rho1[0] * np.sin(d2)

        j22[0] = 3 * self.l_0 * np.cos(a2)
        j22[1] = -8 * rho1[0] * np.sin(b2) * np.sin(c2)
        j22[2] = 4 * rho1[0] * np.cos(d2)

        pos1 = int(not s1 and s2) / \
            (6 * rho2[1]) + int(s1 and s2) / (8 * rho2[2])
        pos2 = int(not s2 and s1) / \
            (6 * rho1[1]) + int(s1 and s2) / (8 * rho1[2])

        J = np.array([[pos1 * np.sum(j11), pos2 * np.sum(j12), int(not (s1 or s2)) * np.cos(q[2]), int(not (s1 or s2)) * (-np.sin(q[2])), 0],
                      [pos1 * np.sum(j21), pos2 * np.sum(j22), int(not (s1 or s2))
                       * np.sin(q[2]), int(not (s1 or s2)) * np.cos(q[2]), 0],
                      [-(int(not s1 and s2) / rho2[1] + int(s1 and s2) * 3 / (4 * rho2[2])), int(
                          not s2 and s1) / rho1[1] + int(s1 and s2) * 3 / (4 * rho1[2]), 0, 0, int(not (s1 or s2))],
                      [int(not s2 and s1) * 3 / (2 * self.l * rho1[0]) + int(s1 and s2) * 3 / (4 * self.l * rho1[2]),
                       int(not s2 and s1) / (self.l * rho1[1]) + int(s1 and s2) * 3 / (4 * self.l * rho1[2]), 0, 0, 0],
                      [int(not s1 and s2) / (self.l * rho2[1]) + int(s1 and s2) * 3 / (4 * self.l * rho2[2]), int(not s1 and s2) * 3 / (2 * self.l * rho2[0]) + int(s1 and s2) * 3 / (4 * self.l * rho2[2]), 0, 0, 0]])

        return J

    def stiffnessPlanner(self, q_d):
        s = [[0, 0], [0, 1], [1, 0], [1, 1]]
        s_array = []
        q_array = []
        q_new = [None] * len(s)

        q = self.q_0
        q_array.append(q)

        diff = np.linalg.norm(q - q_d)

        t = 0.1
        velocity_coeff = 1

        while diff > 10**(-5):
            # print(t)
            q_d_dot = velocity_coeff * (q_d - q) * t
            for i in range(len(s)):
                J = self.hybridJacobian(q, s[i][0], s[i][1])
                upsilon = np.matmul(np.linalg.pinv(J), q_d_dot)
                q_dot = np.matmul(J, upsilon)
                q_new[i] = q + self.dt * q_dot

            error = np.linalg.norm(q_new - q_d, axis=1)
            min_i = np.argmin(error)
            s_array.append(s[min_i])

            q = q_new[min_i]
            q_array.append(q)

            diff = np.linalg.norm(q - q_d)
            t += self.dt

        return s_array, q_array

    def createGraph(self, q_d):
        # Graph-related data structure that contains intermediate configurations
        Q = Queue()
        Q.put(Config(self.q_0, 0.1))

        s = [[0, 0], [0, 1], [1, 0], [1, 1]]
        J = [None] * len(s)
        j = 0

        while not Q.empty():
            config = Q.get()
            q = config.q
            q_d_dot = (q_d - q) * config.t
            print(j)
            j = j + 1

            for i in range(len(s)):
                J = self.hybridJacobian(q, s[i][0], s[i][1])
                upsilon = np.matmul(np.linalg.pinv(J), q_d_dot)
                q_dot = np.matmul(J, upsilon)
                q_new = q + self.dt * q_dot

                if np.linalg.norm(q_new - q) > 0.001 and np.linalg.norm(q_new - q_d) > 0.01:
                    config_new = Config(q_new, config.t + self.dt)
                    Q.put(config_new)


if __name__ == "__main__":

    q_target = np.array([0.01276, -0.01865, 0.6, -61.665, -61.665])
    # q_target = np.array([-0.03, 0.0212, 0.23, -50, 65])
    # q_target = np.array([0.0326, -0.0221, -0.23, -61.665, 61.665])

    control = Control(np.array([0, 0, 0.6, 15, 15]))
    config = control.stiffnessPlanner(q_target)

    # print(config[0])
    # print(len(config[1]))

    anim = graphics.Animation(config[1], q_target)
    anim.plotMotion()

# a = 60 * 10**(-3)
# l_0 = 30 * 10**(-3) # plastic link length
# l = 40 * 10**(-3) # VSS length
# L = 2*l + l_0 # VSB length
# d_bridge = 7 * 10**(-3) # bridge diameter
