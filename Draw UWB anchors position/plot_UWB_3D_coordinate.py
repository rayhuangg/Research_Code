import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def main(exe_name="1.5", anchor_coord_show=False):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho') # orthogonal projection, prevent perspective distortion

    # set grid
    grid_linewidth = 0.5
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 40])
    ax.set_zlim([0, 4])
    ax.xaxis._axinfo["grid"].update({"linewidth": grid_linewidth})
    ax.yaxis._axinfo["grid"].update({"linewidth": grid_linewidth})
    ax.zaxis._axinfo["grid"].update({"linewidth": grid_linewidth})


    labelsize = 22
    ax.set_zticks([0, 1, 2, 3, 4])
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)
    ax.tick_params(axis='z', labelsize=labelsize)


    if exe_name == "1.5":
        uwb_anchor_coords = [
            (0, 0, 1.5), (0, 8, 1.5), (0, 20, 1.5), (0, 32, 1.5), (0, 40, 1.5),
            (10, 0, 1.5), (10, 8, 1.5), (10, 20, 1.5), (10, 32, 1.5), (10, 40, 1.5)
        ]
    elif exe_name == "3.0":
        uwb_anchor_coords = [
            (2, 0, 3), (1, 12, 3), (1, 20, 3), (1, 28, 3), (2, 40, 3),
            (7.6, 0, 3), (8.6, 12, 3), (8.6, 20, 3), (8.6, 28, 3), (7.6, 40, 3)
        ]

    for x, y, z in uwb_anchor_coords:
        ax.scatter(x, y, z, color='royalblue',edgecolors='black', linewidths=2, marker='^', s=600)
        ax.plot([x, x], [y, y], [0, z], color='grey', linestyle='--', linewidth=2)
        if anchor_coord_show:
            ax.text(x-2.2, y+0.5, z-1, f"({x}, {y}, {z})", color='black', fontsize=15)


    # robot track coordinates
    track_coords = [
        (2.2, 4, 0), (2.2, 36, 0), (3.8, 36, 0), (3.8, 4, 0),
        (5.6, 4, 0), (5.6, 36, 0), (7.6, 36, 0), (7.6, 4, 0)
    ]
    xs, ys, zs = zip(*track_coords)
    ax.plot(xs, ys, zs, color='green', marker='o', linewidth=1.5)


    # plot the truth position
    truth_linewidth = 1
    ax.plot([0,0,10,10], [0,0,0,0], [0,3,3,0], color='black', linewidth=truth_linewidth)
    ax.plot([0,0,10,10], [4,4,4,4], [0,3,3,0], color='black', linewidth=truth_linewidth)
    ax.plot([0,0,10,10], [8,8,8,8], [0,3,3,0], color='black', linewidth=truth_linewidth)
    ax.plot([0,0,10,10], [12,12,12,12], [0,3,3,0], color='black', linewidth=truth_linewidth)
    ax.plot([0,0,10,10], [16,16,16,16], [0,3,3,0], color='black', linewidth=truth_linewidth)
    ax.plot([0,0,10,10], [20,20,20,20], [0,3,3,0], color='black', linewidth=truth_linewidth)
    ax.plot([0,0,10,10], [24,24,24,24], [0,3,3,0], color='black', linewidth=truth_linewidth)
    ax.plot([0,0,10,10], [28,28,28,28], [0,3,3,0], color='black', linewidth=truth_linewidth)
    ax.plot([0,0,10,10], [32,32,32,32], [0,3,3,0], color='black', linewidth=truth_linewidth)
    ax.plot([0,0,10,10], [36,36,36,36], [0,3,3,0], color='black', linewidth=truth_linewidth)
    ax.plot([0,0,10,10], [40,40,40,40], [0,3,3,0], color='black', linewidth=truth_linewidth)


    def plot_arc(ax, x, z1, z2, z_peak):
        x_start, x_end = 0, 10
        x_peak = 5

        # calculate the polynomial coefficients
        coeffs = np.polyfit([x_start, x_peak, x_end], [z1, z2, z_peak], 2)
        polynomial = np.poly1d(coeffs)

        # calculate the points on the arc
        xs = np.linspace(x_start, x_end, 50)
        zs = polynomial(xs)

        ax.plot(xs, [x]*len(xs), zs, color='black', linewidth=truth_linewidth )

    plot_arc(ax, 0, 3, 4, 3)
    plot_arc(ax, 4, 3, 4, 3)
    plot_arc(ax, 8, 3, 4, 3)
    plot_arc(ax, 12, 3, 4, 3)
    plot_arc(ax, 16, 3, 4, 3)
    plot_arc(ax, 20, 3, 4, 3)
    plot_arc(ax, 24, 3, 4, 3)
    plot_arc(ax, 28, 3, 4, 3)
    plot_arc(ax, 32, 3, 4, 3)
    plot_arc(ax, 36, 3, 4, 3)
    plot_arc(ax, 40, 3, 4, 3)


    ax.set_box_aspect([10, 40, 4])  # Adjust the aspect ratio
    # ax.legend(['AGV Track', 'UWB Anchor'], loc='upper left')
    ax.view_init(elev=40, azim=315)  # Adjust viewing angle

    plt.savefig(f"UWB_exp{exe_name}.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == '__main__':
    exp_name = "1.5" # "1.5" or "3.0"
    anchor_coord_show = False
    main(exp_name, anchor_coord_show)