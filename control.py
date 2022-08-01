import numpy as np
from queue import Queue
import kinematics
import graphics


class Control:

    def __init__(self, q_0):
        self.q_0 = q_0  # Initial configuration
        self.dt = 0.1

    def stiffnessPlanner(self, q_d):
        s = [[0, 0], [0, 1], [1, 0], [1, 1]]
        s_array = []
        q_array = []
        q_new = [None] * len(s)

        q = self.q_0
        q_array.append(q)

        q_d = np.array(q_d)

        diff = np.linalg.norm(q - q_d)

        t = 0.1
        velocity_coeff = 5

        while diff > 10**(-5):
            # print(t)
            q_d_dot = velocity_coeff * (q_d - q) * t
            for i in range(len(s)):
                J = kinematics.hybridJacobian(self.q_0, q, s[i])
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

    q_start = [0, 0, 0.6, 0, 0]

    # EXAMPLE OF FORWARD KINEMATICS

    sigma = [1, 1]
    v = [0.007, 0.007, 0, 0, 0]
    sim_time = 10
    dt = 0.1
    t = np.arange(dt, sim_time + dt, dt)
    frames = len(t)

    q = kinematics.fk(q_start, sigma, v, sim_time)
    # graphics.plotMotion(q, frames)

    # q_target = [0.01276, -0.01865, 0.6, -61.665, -61.665]
    # q_target = [-0.03, 0.0212, 0.23, -50, 65]
    # q_target = [0.0326, -0.0221, -0.23, -61.665, 61.665]
    q_target = q[-1].tolist()

    control = Control(q_start)
    config = control.stiffnessPlanner(q_target)
    frames = len(config[1])

    print(config[0])

    graphics.plotMotion(config[1], frames, q_d=q_target)
