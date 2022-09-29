import numpy as np
import kinematics
import graphics
import random as rnd
import globals_
import pandas as pd


class Control:

    def __init__(self, q_0):
        self.q_0 = q_0  # initial configuration

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

        t = globals_.DT  # current time
        # feedback gain
        velocity_coeff = np.ones((5,), dtype=int)
        # Index of the current stiffness configuration
        current_i = None

        while dist > 0:

            flag = False  # indicates whether VSB stiffness has changed

            # INVERSE KINEMATICS

            q_tilda = 1 * (q_t - q) * t
            for i in range(len(s)):
                # Jacobian matrix
                J = kinematics.hybridJacobian(self.q_0, q, s[i])
                # velocity input commands
                v_[i] = np.matmul(np.linalg.pinv(J), q_tilda)
                q_dot = np.matmul(J, v_[i])
                q_[i] = q + (1 - np.exp(-1 * t)) * q_dot * globals_.DT

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
            # print(s_list[current_i])

            if flag:
                switch_counter += 1

            t += globals_.DT  # increment time

        return q_list, s_list, switch_counter


def phase_transition(s1, s2):
    if s1 == 0:
        return s2
    if s2 == 0:
        return s1
    return s1 * s2


if __name__ == "__main__":

    # SIMULATION PARAMETERS

    sim_time = 10  # simulation time
    t = np.arange(globals_.DT, sim_time + globals_.DT, globals_.DT)  # span

    # Initial configuration
    q_start = [0, 0, rnd.uniform(-0.6, 0.06),
               rnd.uniform(-60.0, 60.0), rnd.uniform(-60.0, 60.0)]
    # print("Start: ", q_start)
    # q_start = [0.26,  0.23, -0.3, 28, -16]
    # q_start = [0.29, 0.2, -0.62, -27, 15]
    # q_start = [0.231, 0.20, 0.44, -11, -13]
    # q_start = [0.238, 0.276, -0.24, -38, -24]
    # q_start = [0.231,  0.262, -0.34, -27, -16]

    # FORWARD KINEMATICS

    # Stiffness of the VS segments
    sigma = [rnd.randint(0, 1), rnd.randint(0, 1)]
    # print("Stiffness: ", sigma)
    # Input velocity commands
    v = [rnd.uniform(-0.008, 0.008), rnd.uniform(-0.008, 0.008),
         rnd.uniform(-0.03, 0.03), rnd.uniform(-0.03, 0.03), rnd.uniform(-0.1, 0.1)]
    # print("Velocity: ", v)

    # Generate a trajectory by an FK model
    q_list = kinematics.fk(q_start, sigma, v, sim_time)
    frames = len(q_list)  # number of frames
    # print(len(q_list))

    # Animation of the 2SRR motion along the trajectory
    # graphics.plotMotion(q_list, frames)

    # MOTION PLANNER (STIFFNESS PLANNER + INVERSE KINEMATICS)

    # We take the last configuration of an FK trajectory
    # # as a target configuration
    q_target = q_list[-1].tolist()
    # print("Target: ", q_target)
    # q_target = [0.24, 0.21, 0.5, 28, 3]
    # q_target = [0.27, 0.24, 0.9, 30, 15]
    # q_target = [0.225, 0.245, -0.235, 38, 34]
    # q_target = [0.24,  0.3, -0.15, -15, -5]
    # q_target = [0.241, 0.266, -0.5, -25, 12]

    # Initialize the controller
    control = Control(q_start)
    # Generate a trajectory and a sequence of stiffness values
    config = control.motionPlanner(q_target)
    # for i in range(15):
    #     config[0].append(config[0][-1])
    #     config[1].append(config[1][-1])

    frames = len(config[0])
    # print(frames)

    # print("Stiffness transitions: ", config[2])
    # print(config[0], config[1])
    df = pd.DataFrame(config[0], columns=['x', 'y', 'phi', 'k1', 'k2'])
    df = pd.concat([df, pd.DataFrame(config[1], columns=['s1', 's2'])], axis=1)
    tr = pd.concat([df.s1.diff().fillna(0), df.s2.diff().fillna(0)], axis=1)
    tr_comb = tr.apply(
        lambda x: phase_transition(x['s1'], x['s2']), axis=1)

    times = [1] * len(df)

    for i in tr_comb.index[tr_comb == 1].tolist():
        times[i] = 30
    for i in tr_comb.index[tr_comb == -1].tolist():
        times[i] = 90

    # df = df.loc[df.index.repeat(times)].reset_index(drop=True)
    # df.to_csv('Data/simulation2csv', index=False)

    # Animation of the 2SRR motion towards the target
    # graphics.plotMotion(config[0], config[1], frames, q_t=q_target)
    graphics.plotAnalysis(q_list, sigma, config[0], config[1][1:])




