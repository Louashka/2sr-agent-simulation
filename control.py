import numpy as np
from queue import Queue
import kinematics
import graphics


class Control:

    def __init__(self, q_0):
        self.q_0 = q_0  # Initial configuration

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


if __name__ == "__main__":

    # q_target = np.array([0.01276, -0.01865, 0.6, -61.665, -61.665])
    # q_target = np.array([-0.03, 0.0212, 0.23, -50, 65])
    # q_target = np.array([0.0326, -0.0221, -0.23, -61.665, 61.665])

    # control = Control(np.array([0, 0, 0.6, 15, 15]))
    # config = control.stiffnessPlanner(q_target)

    q_start = [0, 0, 0, 0, 0]
    sigma = [1, 0]
    v = [-0.005, 0.00, 0, 0, 0]
    sim_time = 10

    q = kinematics.fk(q_start, sigma, v, sim_time)
    # val = kinematics.hybridJacobian(q_start, q_start, sigma)
    # print(val)

    anim = graphics.Animation(q)
    anim.plotMotion()
