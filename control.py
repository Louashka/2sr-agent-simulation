import numpy as np
from queue import Queue
import kinematics
import graphics
import random as rnd


class Control:

    def __init__(self, q_0):
        self.q_0 = q_0  # Initial configuration
        self.dt = 0.1

    def stiffnessPlanner(self, q_d):
        s = [[0, 0], [0, 1], [1, 0], [1, 1]]
        s_array = []
        s_array.append(s[0])
        switch_counter = 0

        q_array = []
        q_new = [None] * len(s)

        q = self.q_0
        q_array.append(q)

        q_d = np.array(q_d)

        diff = np.linalg.norm(q - q_d)
        error_array = []

        t = 0.1
        velocity_coeff = np.ones((5,), dtype=int)
        current_i = None

        while diff > 0:

            flag = False

            q_tilda = velocity_coeff * (q_d - q) * t
            for i in range(len(s)):
                J = kinematics.hybridJacobian(self.q_0, q, s[i])
                upsilon = np.matmul(np.linalg.pinv(J), q_tilda)
                q_dot = np.matmul(J, upsilon)
                q_new[i] = q + (1 - np.exp(-1 * t)) * q_dot * self.dt

            error = np.linalg.norm(q_new - q_d, axis=1)
            min_i = np.argmin(error)

            step = np.linalg.norm(q - np.array(q_new), axis=1)

            if min_i != current_i and current_i is not None:
                if step[current_i] > 10**(-17):
                    min_i = current_i
                else:
                    flag = True

            q = q_new[min_i]

            if (step[min_i] > 10 ** (-5)):
                q_array.append(q)
                s_array.append(s[min_i])

                if flag:
                    switch_counter += 1

            error_array.append(step[min_i])

            current_i = min_i

            diff = np.linalg.norm(q - q_d)
            t += self.dt

        return q_array, s_array, switch_counter, error_array


if __name__ == "__main__":

    q_start = [0, 0, rnd.uniform(-0.6, 0.06),
               rnd.uniform(-80.0, 80.0), rnd.uniform(-80.0, 80.0)]

    # EXAMPLE OF FORWARD KINEMATICS

    sigma = [rnd.randint(0, 1), rnd.randint(0, 1)]
    print("Stiffness: ", sigma)
    v = [rnd.uniform(-0.008, 0.008), rnd.uniform(-0.008, 0.008),
         rnd.uniform(-0.03, 0.03), rnd.uniform(-0.03, 0.03), rnd.uniform(-0.1, 0.1)]
    print("Velocity: ", v)

    sigma = [1, 1]
    v = [0.004, 0.007, 0, 0, 0]
    sim_time = 10
    dt = 0.1
    t = np.arange(dt, sim_time + dt, dt)
    frames = len(t)

    q = kinematics.fk(q_start, sigma, v, sim_time)
    # graphics.plotMotion(q, frames)

    q_target = q[-1].tolist()

    control = Control(q_start)
    config = control.stiffnessPlanner(q_target)
    frames = len(config[0])

    # print(config[1])
    print("Stiffness transitions: ", config[2])

    graphics.plotMotion(config[0], config[1], frames, q_d=q_target)
