import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import globals_


link_width = globals_.D_BRIDGE
link_length = globals_.L_LINK

LINK_DIAG = ((link_length / 2)**2 + (link_width / 2)**2)**(1 / 2)

font_size = 22
fig, ax = plt.subplots()
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)

q_list = []
q_target = []
s_array = []

x_range = 0
y_range = 0

link = Rectangle((0, 0), 0, 0, fc='y')
arc1, = ax.plot([], [], lw=5, color="blue")
arc2, = ax.plot([], [], lw=5, color="blue")
centre, = ax.plot([], [], lw=5, marker=".", color="black")

stiffness_text = ax.text(0, 0, '', fontsize=font_size)

target_link = Rectangle((0, 0), 0, 0, fc='y', alpha=0.3)
target_arc1, = ax.plot([], [], lw=3, color="black", alpha=0.3)
target_arc2, = ax.plot([], [], lw=3, color="black", alpha=0.3)

# ax.axis('off')


def init():
    global ax, x_range, y_range

    x_range, y_range = defineRange()
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")

    ax.add_patch(link)

    if q_target:
        ax.add_patch(target_link)


def defineRange():
    margin = 0.058

    q_array = np.array(q_list)
    x_min, y_min = q_array[:, :2].min(axis=0)
    x_max, y_max = q_array[:, :2].max(axis=0)

    ax_range = max(x_max - x_min, y_max - y_min) + margin

    x_range = (x_min - margin, x_min + ax_range)
    y_range = (y_min - margin, y_min + ax_range)

    return x_range, y_range


def genArc(q, seg):
    s = np.linspace(0, globals_.L_VSS, 50)

    flag = -1 if seg == 1 else 1

    gamma_array = q[2] + flag * q[2 + seg] * s

    x_0 = q[0] + flag * np.cos(q[2]) * link_length / 2
    y_0 = q[1] + flag * np.sin(q[2]) * link_length / 2

    if q[2 + seg] == 0:
        x = x_0 + [0, flag * globals_.L_VSS * np.cos(q[2])]
        y = y_0 + [0, flag * globals_.L_VSS * np.sin(q[2])]
    else:
        x = x_0 + np.sin(gamma_array) / \
            q[2 + seg] - np.sin(q[2]) / q[2 + seg]
        y = y_0 - np.cos(gamma_array) / \
            q[2 + seg] + np.cos(q[2]) / q[2 + seg]

    return [x, y]


def update(i):
    global link, target_link, arc1, arc2, centre, stiffness_text, target_arc1, target_arc2
    q = q_list[i]

    x = q[0]
    y = q[1]
    phi = q[2]

    x0 = x - link_length / 2
    y0 = y - link_width / 2

    link.set_width(link_length)
    link.set_height(link_width)
    link.set_xy([x0, y0])

    transform = mpl.transforms.Affine2D().rotate_around(
        x, y, phi) + ax.transData
    link.set_transform(transform)

    seg1 = genArc(q, 1)
    seg2 = genArc(q, 2)

    arc1.set_data(seg1[0], seg1[1])
    arc2.set_data(seg2[0], seg2[1])

    if s_array[i][0] == 0:
        arc1.set_color("blue")
    else:
        arc1.set_color("red")

    if s_array[i][1] == 0:
        arc2.set_color("blue")
    else:
        arc2.set_color("red")

    centre.set_data(x, y)

    # stiffness_text.set_text(
    #     "s1: " + str(s_array[i][0]) + ", s2: " + str(s_array[i][1]))
    # stiffness_text.set_position(
    #     (x_range[1] - (x_range[1] - x_range[0]) / 3, y_range[1] - (y_range[1] - y_range[0]) / 15))
    # stiffness_text.set_position(
    #     (x_range[0] + (x_range[1] - x_range[0]) / 25, y_range[0] + (y_range[1] - y_range[0]) / 40))

    if q_target:

        x_t = q_target[0] - link_length / 2
        y_t = q_target[1] - link_width / 2
        phi_t = q_target[2]

        target_link.set_width(link_length)
        target_link.set_height(link_width)
        target_link.set_xy([x_t, y_t])

        target_transform = mpl.transforms.Affine2D().rotate_around(
            q_target[0], q_target[1], phi_t) + ax.transData
        target_link.set_transform(target_transform)

        target_seg1 = genArc(q_target, 1)
        target_seg2 = genArc(q_target, 2)

        target_arc1.set_data(target_seg1[0], target_seg1[1])
        target_arc2.set_data(target_seg2[0], target_seg2[1])

        return link, arc1, arc2, centre, stiffness_text, target_link, target_arc1, target_arc2,

    return link, arc1, arc2, centre, stiffness_text,


def plotMotion(q, s, frames, q_t=[]):
    global q_list, s_array, q_target
    q_list = q
    s_array = s
    q_target = q_t

    anim = FuncAnimation(fig, update, frames,
                         init_func=init, interval=1, repeat=True)

    # Save animation
    mywriter = FFMpegWriter(fps=30)
    # anim.save('Animation/sim_for_video_5.mp4', writer=mywriter, dpi=300)

    plt.show()

def plotAnalysis(q_fk, s_fk, q_pm, s_pm):

    q_target = q_fk[-1].tolist()
    colours = ['#0072bd', '#d95319', '#edb120', '#008176']
    s_ref = [[0, 0], [0, 1], [1, 0], [1, 1]]

    cols = ['x', 'y', 'phi', 'k1', 'k2']
    pd_fk = pd.DataFrame(data = q_fk, columns = cols)
    pd_mp = pd.DataFrame(data = q_pm, columns = cols)

    error_fk = np.sqrt(np.square(pd_fk - q_target).sum(axis=1)).values
    error_fk = error_fk / max(error_fk)
    t_norm_fk = np.linspace(0, 1, num=len(error_fk), endpoint=True)

    error_mp = np.sqrt(np.square(pd_mp - q_target).sum(axis=1)).values
    error_mp = error_mp / max(error_mp)
    t_norm_mp = np.linspace(0, 1, num=len(error_mp), endpoint=True)

    fig = plt.figure(figsize=(12,4.5))

    plt.plot(t_norm_fk, error_fk, '--', color = colours[s_ref.index(s_fk)], linewidth = 7, label = 'FK ' + str(s_fk))

    s_set = []
    i_set = []

    current_s = s_pm[0]
    s_set.append(current_s)
    i_set.append(0)

    counter = 0
    for s in s_pm:
        if s != current_s:
            s_set.append(s)
            i_set.append(counter)

            current_s = s

        counter += 1

    if i_set[-1] != len(s_pm)-1:
        i_set.append(len(s_pm))

    print(i_set)

    for i, s in zip(range(len(i_set)-1), s_set):
        print(s)
        print(colours[s_ref.index(s)])
        error_slice = error_mp[i_set[i]:i_set[i+1]]
        t_slice = t_norm_mp[i_set[i]:i_set[i+1]]
        plt.plot(t_slice, error_slice, color = colours[s_ref.index(s)], linewidth = 7, label = 'MP ' + str(s))

    plt.xlabel('Normalised Time', fontsize = font_size)
    plt.ylabel('Normalised ' + r'$\Delta q$', fontsize = font_size)
    plt.legend(ncol = 2, fontsize = font_size, handlelength = 2)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.grid()

    fig.tight_layout()
    fig.savefig('Plots/fk-vs-mp-13.png', format='png', dpi=300)
    plt.show()
