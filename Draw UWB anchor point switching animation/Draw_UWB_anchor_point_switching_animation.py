import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import Patch


def read_file(filename):
    data_list = []
    with open(filename) as file:
        for line in file:
            line = line.strip()  # Remove newline characters and whitespace
            if line:
                elements = line.split(", ")  # Split the line by comma and space
                data = ",".join(elements[1:])  # Join the remaining elements as a string
                data_list.append(data)
    return data_list


def uwb_rawdata_to_json(data):
    data = data.split(",")
    anchor_number = int(data[data.index("DIST") + 1])  # number of receive anchor, 3 or 4
    uwb_data_dict = dict(Anchor_number=anchor_number,
                            AN0={},
                            AN1={},
                            AN2={},
                            AN3={},
                            POS={},
                            )

    for i in range(anchor_number):
        uwb_data_dict[f"AN{i}"]["id"] = data[data.index("AN" + str(i)) + 1]
        uwb_data_dict[f"AN{i}"]["x"] = data[data.index("AN" + str(i)) + 2]
        uwb_data_dict[f"AN{i}"]["y"] = data[data.index("AN" + str(i)) + 3]
        uwb_data_dict[f"AN{i}"]["z"] = data[data.index("AN" + str(i)) + 4]
        uwb_data_dict[f"AN{i}"]["dist"] = data[data.index("AN" + str(i)) + 5]

    # Save the accurate position information
    if "POS" in data:
        pos_x = data[data.index("POS")+1]
        pos_y = data[data.index("POS")+2]
        uwb_data_dict["POS"]["x"] = pos_x
        uwb_data_dict["POS"]["y"] = pos_y
        uwb_data_dict["POS"]["z"] = data[data.index("POS")+3]
        uwb_data_dict["POS"]["quality_factor"] = data[data.index("POS")+4]

        uwb_all_data = json.dumps(uwb_data_dict)

    else:
        uwb_all_data = json.dumps(uwb_data_dict)
    return uwb_all_data


def update(frame):
    x1 = uwb1_pos_x_all[:frame+1]
    y1 = uwb1_pos_y_all[:frame+1]
    # x1 = uwb1_pos_x_all[frame]  # single point
    # y1 = uwb1_pos_y_all[frame]
    line.set_data(x1, y1)

    x2 = uwb2_pos_x_all[:frame+1]
    y2 = uwb2_pos_y_all[:frame+1]
    # x2 = uwb2_pos_x_all[frame]
    # y2 = uwb2_pos_y_all[frame]
    line2.set_data(x2, y2)

    scatter_points = []
    max_num_anchors = max(int(uwb1_data_list[frame]["Anchor_number"]), int(uwb2_data_list[frame]["Anchor_number"]))
    patch_points = []

    for i in range(max_num_anchors):
        if i < int(uwb1_data_list[frame]["Anchor_number"]):
            anchor1 = uwb1_data_list[frame][f"AN{i}"]
            an1_x = float(anchor1["x"])
            an1_y = float(anchor1["y"])
            scatter1 = ax.scatter(an1_x, an1_y, color='red', marker='o', s=150, alpha=0.8, edgecolor='black')
            scatter_points.append(scatter1)
            circle1 = ax.add_patch(plt.Circle((an1_x, an1_y), radius=15, facecolor='lightblue', edgecolor='red', linewidth=1, fill=True, alpha=0.1))
            # circle1.set_clip_on(False)
            patch_points.append(circle1)


        if i < int(uwb2_data_list[frame]["Anchor_number"]):
            anchor2 = uwb2_data_list[frame][f"AN{i}"]
            an2_x = float(anchor2["x"])
            an2_y = float(anchor2["y"])
            scatter2 = ax.scatter(an2_x, an2_y, color='blue', marker='*', s=180, alpha=0.8, edgecolor='black')
            scatter_points.append(scatter2)
            circle2 = ax.add_patch(plt.Circle((an2_x, an2_y), radius=15, facecolor='lightblue', edgecolor='blue', linewidth=1, fill=True, alpha=0.1))
            # circle2.set_clip_on(False)
            patch_points.append(circle2)

    return line, line2, *patch_points, *scatter_points,


def init():
    line.set_data([], [])
    line2.set_data([], [])

    # 標記anchor位置
    for i, _ in enumerate(anchor_x):
        plt.scatter(anchor_x[i], anchor_y[i], color='gray', s=50)
        # 標記anchor座標，截圖用
        # plt.annotate(f"{anchor_x[i],anchor_y[i]}", (anchor_x[i]+0.5,anchor_y[i]+0.5))

        # circle = patches.Circle((anchor_x[i], anchor_y[i]), radius=15, facecolor='lightblue', edgecolor='gray', linewidth=1, fill=True, alpha=0.05)
        # circle.set_clip_on(False) # 要不要裁減超過座標軸的區域
        # ax.add_patch(circle)

    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='UWB1-AN'),
        Patch(facecolor='blue', edgecolor='black', label='UWB2-AN')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    return line,


date = "day2"

uwb1_path_day1 = "2023_07_04-17_18_41/uwb_data/uwb1_data.txt"
uwb1_path_day2 = "2023_07_05-11_08_30/uwb_data/uwb1_data.txt"
uwb2_path_day1 = "2023_07_04-17_18_41/uwb_data/uwb2_data.txt"
uwb2_path_day2 = "2023_07_05-11_08_30/uwb_data/uwb2_data.txt"


# =========== Day1 ===========
anchor_x_day1 = [2.05, 2.05, 4.83, 4.83, 4.83, 7.6, 7.6, 9.6, 0]
anchor_y_day1 = [12, 28, 4, 20, 36, 12, 28, 40, 0]
# =========== Day2 ===========
anchor_x_day2 = [7.6, 8.6, 8.6, 8.6, 7.6, 2.05, 1, 1, 1, 2.05]
anchor_y_day2 = [40, 28, 20, 12, 0, 0, 12, 20, 28, 40]


if date == "day1":
    uwb1_path = uwb1_path_day1
    uwb2_path = uwb2_path_day1
    anchor_x = anchor_x_day1
    anchor_y = anchor_y_day1

elif date == "day2":
    uwb1_path = uwb1_path_day2
    uwb2_path = uwb2_path_day2
    anchor_x = anchor_x_day2
    anchor_y = anchor_y_day2

uwb1_raw_data = read_file(uwb1_path)
uwb2_raw_data = read_file(uwb2_path)
uwb1_pos_x_all = []
uwb1_pos_y_all = []
uwb1_data_list = []
uwb2_pos_x_all = []
uwb2_pos_y_all = []
uwb2_data_list = []

for data in uwb1_raw_data:
    json_data = uwb_rawdata_to_json(data)
    json_data = json.loads(json_data)
    uwb1_data_list.append(json_data)

    pos_x = json_data["POS"]["x"]
    pos_y = json_data["POS"]["y"]
    uwb1_pos_x_all.append(float(pos_x))
    uwb1_pos_y_all.append(float(pos_y))

for data in uwb2_raw_data:
    json_data = uwb_rawdata_to_json(data)
    json_data = json.loads(json_data)
    uwb2_data_list.append(json_data)

    pos_x = json_data["POS"]["x"]
    pos_y = json_data["POS"]["y"]
    uwb2_pos_x_all.append(float(pos_x))
    uwb2_pos_y_all.append(float(pos_y))


fig = plt.figure(figsize=(6, 9))
ax = plt.axes(xlim=(-2, 12), ylim=(-2, 42))
ax.set_aspect('equal') # set qequal aspect ratio
line, = ax.plot([], [], 'o', markersize=1)
line2, = ax.plot([], [], 'o', markersize=1)



ani = animation.FuncAnimation(fig, update, frames=len(uwb1_pos_x_all), interval=10, init_func=init, blit=True)
# ani.save('animation.mp4', fps=30)
plt.show()