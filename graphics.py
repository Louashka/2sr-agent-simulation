import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import numpy as np


class Animation:

    def __init__(self, q_list):

        self.L_VSS = 40 * 10**(-3)  # VSS length
        self.L0 = 30 * 10**(-3)  # plastic link length
        self.D_BRIDGE = 7 * 10**(-3)  # bridge diameter
        self.LINK_DIAG = ((self.L0 / 2)**2 + (self.D_BRIDGE / 2)**2)**(1 / 2)

        self.q_list = q_list
        self.n = len(self.q_list)

        x_range, y_range = self.defineRange()

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=x_range, ylim=y_range)
        self.ax.set_aspect("equal")
        # self.path, = self.ax.plot([], [], lw=2, marker="o")

        self.patch = Rectangle((0, 0), 0, 0, fc='y')
        self.ax.add_patch(self.patch)

        self.arc1, = self.ax.plot([], [], lw=2, marker=".", color="blue")
        self.arc2, = self.ax.plot([], [], lw=2, marker=".", color="blue")

        self.centre, = self.ax.plot([], [], lw=2, marker=".", color="black")

    def defineRange(self):
        margin = 0.1

        q_array = np.array(self.q_list)
        x_min, y_min = q_array[:, :2].min(axis=0)
        x_max, y_max = q_array[:, :2].max(axis=0)

        ax_range = max(x_max - x_min, y_max - y_min) + margin

        x_range = (x_min - margin, x_min + ax_range)
        y_range = (y_min - margin, y_min + ax_range)

        return x_range, y_range

    def genArc(self, q, seg):
        s = np.linspace(0, self.L_VSS, 50)

        flag = -1 if seg == 1 else 1

        gamma_array = q[2] + flag * q[2 + seg] * s

        x_0 = q[0] + flag * np.cos(q[2]) * self.L0 / 2
        y_0 = q[1] + flag * np.sin(q[2]) * self.L0 / 2

        if q[2 + seg] == 0:
            x = x_0 + [0, flag * self.L_VSS * np.cos(q[2])]
            y = y_0 + [0, flag * self.L_VSS * np.sin(q[2])]
        else:
            x = x_0 + np.sin(gamma_array) / \
                q[2 + seg] - np.sin(q[2]) / q[2 + seg]
            y = y_0 - np.cos(gamma_array) / \
                q[2 + seg] + np.cos(q[2]) / q[2 + seg]

        return [x, y]

    def update(self, i):
        q = self.q_list[i]

        x = q[0]
        y = q[1]
        phi = q[2]

        x0 = x - self.L0 / 2
        y0 = y - self.D_BRIDGE / 2

        self.patch.set_width(self.L0)
        self.patch.set_height(self.D_BRIDGE)
        self.patch.set_xy([x0, y0])

        transform = mpl.transforms.Affine2D().rotate_around(
            x, y, phi) + self.ax.transData
        self.patch.set_transform(transform)

        seg1 = self.genArc(q, 1)
        seg2 = self.genArc(q, 2)

        self.arc1.set_data(seg1[0], seg1[1])
        self.arc2.set_data(seg2[0], seg2[1])

        self.centre.set_data(x, y)

        return self.patch, self.arc1, self.arc2, self.centre

    def plotMotion(self):
        anim = FuncAnimation(self.fig, self.update,
                             frames=self.n, interval=1, repeat=True)
        plt.show()
