import numpy as np
import kinematics
import graphics
import random as rnd


class Control:

    def __init__(self, q_0):
        self.q_0 = q_0  # initial configuration
        self.dt = 0.1  # step size

    def motionPlanner(self, q_target):
        # A set of possible stiffness configurations
        s = [[0, 0], [0, 1], [1, 0], [1, 1]]
        # Initialize a sequence of VSB stiffness values
        s_list = []
        # 2SRR always starts from the rigid state
        s_list.append(s[0])
        # Initialize the number of stiffness transitions
        switch_counter = 0

        # Initialize a trajectory
        q_list = []
        q = self.q_0  # current configuration
        q_list.append(q)

        # A set of possible configurations
        q_ = [None] * len(s)
        v_ = [None] * len(s)

        q_t = np.array(q_target)
        # Euclidean distance between current and target configurations (error)
        dist = np.linalg.norm(q - q_t)

        t = 0.1  # current time
        # feedback gain
        velocity_coeff = np.ones((5,), dtype=int)
        # Index of the current stiffness configuration
        current_i = None

        while dist > 0:

            flag = False  # indicates whether VSB stiffness has changed

            # INVERSE KINEMATICS

            q_tilda = velocity_coeff * (q_t - q) * t
            for i in range(len(s)):
                # Jacobian matrix
                J = kinematics.hybridJacobian(self.q_0, q, s[i])
                # velocity input commands
                v_[i] = np.matmul(np.linalg.pinv(J), q_tilda)
                q_dot = np.matmul(J, v_[i])
                q_[i] = q + (1 - np.exp(-1 * t)) * q_dot * self.dt

            # Determine the stiffness configuration that promotes
            # faster approach to the target
            dist_ = np.linalg.norm(q_ - q_t, axis=1)
            min_i = np.argmin(dist_)

            # The extent of the configuration change
            delta_q_ = np.linalg.norm(q - np.array(q_), axis=1)

            # Stiffness transition is committed only if the previous
            # stiffness configuration does not promote further motion
            if min_i != current_i and current_i is not None:
                if delta_q_[current_i] > 10**(-17):
                    min_i = current_i
                else:
                    flag = True

            q = q_[min_i]  # update current configuration
            dist = np.linalg.norm(q - q_t)  # update error
            current_i = min_i  # update current stiffness

            if (delta_q_[current_i] > 10 ** (-5)):
                q_list.append(q)
                s_list.append(s[current_i])
                print(v_[current_i])

                if flag:
                    switch_counter += 1

            t += self.dt  # increment time

        return q_list, s_list, switch_counter


if __name__ == "__main__":

    # SIMULATION PARAMETERS

    sim_time = 10  # simulation time
    dt = 0.1  # step size
    t = np.arange(dt, sim_time + dt, dt)  # span
    frames = len(t)  # number of frames

    # Initial configuration
    q_start = [0, 0, rnd.uniform(-0.6, 0.06),
               rnd.uniform(-80.0, 80.0), rnd.uniform(-80.0, 80.0)]
  #   q_start = [1.27296588e-01, -2.48691099e-02,  7.27445334e-02,  2.52447516e+01,
  # 2.26051360e+01]

    # FORWARD KINEMATICS

    # Stiffness of the VS segments
    sigma = [rnd.randint(0, 1), rnd.randint(0, 1)]
    print("Stiffness: ", sigma)
    # Input velocity commands
    v = [rnd.uniform(-0.008, 0.008), rnd.uniform(-0.008, 0.008),
         rnd.uniform(-0.03, 0.03), rnd.uniform(-0.03, 0.03), rnd.uniform(-0.1, 0.1)]
    print("Velocity: ", v)

    # Generate a trajectory by an FK model
    q_list = kinematics.fk(q_start, sigma, v, sim_time)

    # Animation of the 2SRR motion along the trajectory
    # graphics.plotMotion(q_list, frames)

    # MOTION PLANNER (STIFFNESS PLANNER + INVERSE KINEMATICS)

    # We take the last configuration of an FK trajectory
    # as a target configuration
    q_target = q_list[-1].tolist()
    # q_target = [0.30314961,  0.27486911, -1.13779664, 22.76592851, 24.13558513]

    # Initialize the controller
    control = Control(q_start)
    # Generate a trajectory and a sequence of stiffness values
    config = control.motionPlanner(q_target)
    frames = len(config[0])

    # print("Stiffness transitions: ", config[2])
    # print(config[0])

    # Animation of the 2SRR motion towards the target
    graphics.plotMotion(config[0], config[1], frames, q_t=q_target)
