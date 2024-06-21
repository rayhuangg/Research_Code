from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import yaml
import argparse
from datetime import datetime

from utils.extract_frame import extract_frame

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    if '_BASE_' in config:
        base_path = config.pop('_BASE_')
        base_config = load_config(os.path.join(os.path.dirname(config_path), base_path))
        base_config.update(config)
        return base_config
    return config

# Argument parser to get the config file path
parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Accessing the parameters from the config
mode = config.get('mode', 'video')
video_path = config.get('video_path', 'Data/exp_lab2.mp4')
image_path = config.get('image_path', 'Data/imgs/exp_lab2/frame_0004.jpg')
folder_path = config.get('folder_path', 'Data/imgs/exp_lab2')
calib_file_path = config.get('calib_file_path', 'calibration_images/exp_lab2/camera_calibration.npz')
output_folder = config.get('output_folder', f'Data/robot_offset_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
sampling_interval = config.get('sampling_interval', 2)
car_corners = config.get('car_corners', [(790, 295), (1151, 307), (763, 771), (1151, 764)])
car_front_center = config.get('car_front_center', (966, 304))
mm_to_pixel_ratio = config.get('mm_to_pixel_ratio', 1.416)
pixel_to_mm_ratio = 1 / mm_to_pixel_ratio
display_realtime = config.get('display_realtime', True)
use_calibration = config.get('use_calibration', False)
debug_frame = config.get('debug_frame', False)
start_recognition_time = config.get('start_recognition_time', {"start_min": 0, "start_sec": 0, "start_ms": 0.0})
manual_annote_missing_frame = config.get('manual_annote_missing_frame', False)

# Print loaded parameters for verification
print("Mode:", mode)
print("Video Path:", video_path)
print("Image Path:", image_path)
print("Folder Path:", folder_path)
print("Calibration File Path:", calib_file_path)
print("Output Folder:", output_folder)
print("Sampling Interval:", sampling_interval)
print("Car Corners:", car_corners)
print("Car Front Center:", car_front_center)
print("mm to Pixel Ratio:", mm_to_pixel_ratio)
print("Pixel to mm Ratio:", pixel_to_mm_ratio)
print("Display Realtime:", display_realtime)
print("Use Calibration:", use_calibration)
print("Debug Frame:", debug_frame)
print("Start Recognition Time:", start_recognition_time)
print("Manual Annotate Missing Frame:", manual_annote_missing_frame)


results = []

# Load camera calibration parameters
if use_calibration:
    try:
        with np.load(calib_file_path) as data:
            camera_matrix = data["camera_matrix"]
            dist_coeffs = data["dist_coeffs"]
    except FileNotFoundError:
        print(f"Error: Camera calibration file not found at {calib_file_path}.")
        camera_matrix = None
        dist_coeffs = None
    except Exception as e:
        print(f"Error: An error occurred while loading camera calibration file: {e}")
        camera_matrix = None
        dist_coeffs = None
else:
    camera_matrix = None
    dist_coeffs = None

def undistort_image(image):
    if camera_matrix is not None and dist_coeffs is not None:
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        x, y, w, h = roi
        # Ensure ROI is within image dimensions
        # if w > 0 and h > 0:
        #     undistorted_img = undistorted_img[y:y+h, x:x+w]
        return undistorted_img
    return image

def detect_white_lines(frame, search_area):
    """
    Detects white lines in a given frame within a specified search area.

    Args:
        frame (numpy.ndarray): The input frame in BGR format.
        search_area (tuple): The coordinates of the search area in the format (x1, y1, x2, y2).

    Returns:
        white_lines (list): A list of tuples representing the detected white lines in the format (lx1, ly1, lx2, ly2).
        binary (numpy.ndarray): The binary image after morphological operations.

    """
    x1, y1, x2, y2 = search_area
    search_frame = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([85, 0, 109])
    upper_white = np.array([180, 199, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Morphological operations to reduce noise
    kernel = np.ones((10, 10), np.uint8)  # Define the kernel size for morphological operations
    binary = cv2.dilate(mask, kernel, iterations=3)   # Dilation operation
    binary = cv2.erode(binary, kernel, iterations=3)  # Erosion operation

    edges = cv2.Canny(binary, 30, 90, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=80, maxLineGap=50)
    white_lines = []
    if lines is not None:
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]  # Coordinates within the ROI
            lx1 += x1  # Convert coordinates to original image coordinates
            ly1 += y1
            lx2 += x1
            ly2 += y1

            # Exclude lines on the boundaries of the search area
            offset = 3
            if (x1 + offset < lx1 < x2 - offset) and (x1 + offset < lx2 < x2 - offset) and (y1 + offset < ly1 < y2 - offset) and (y1 + offset < ly2 < y2 - offset):
                white_lines.append((lx1, ly1, lx2, ly2))
                # Visualize the detected white lines
                cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)

            white_lines.sort(key=lambda x: min(x[0], x[3]))  # Sort by the smaller value between ly1 and ly2
    # cv2.imshow("Edges", edges)
    return white_lines, binary

def calculate_line_length(point):
    x1, y1, x2, y2 = point
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return length

def calculate_center_line(corners):
    (x1, y1), (x2, y2) = corners
    center_line = ((x1 + x2) / 2, (y1 + y2) / 2)
    return center_line

def extend_line_to_border(line, car_corners):
    """
    Extend the line and calculate its intersection points with the car_corners.

    Args:
        line (tuple): The line segment represented by (x1, y1, x2, y2).
        car_corners (list): The corners of the car in the format [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].

    Returns:
        list: The intersection points with the car boundaries.
    """
    x1, y1, x2, y2 = line
    intersections = []

    # Calculate line equation y = mx + c
    if x2 - x1 != 0:
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
    else:
        m = None  # Vertical line

    # Define the car boundaries
    left, right = min(car_corners[0][0], car_corners[3][0]), max(car_corners[1][0], car_corners[2][0])
    top, bottom = min(car_corners[0][1], car_corners[1][1]), max(car_corners[2][1], car_corners[3][1])

    # Check intersection with the left boundary (x = left)
    if m is not None:
        y_left = int(m * left + c)
        if top <= y_left <= bottom:
            intersections.append((left, y_left))

    # Check intersection with the right boundary (x = right)
    if m is not None:
        y_right = int(m * right + c)
        if top <= y_right <= bottom:
            intersections.append((right, y_right))

    # Check intersection with the top boundary (y = top)
    if m is not None:
        if m != 0:  # Avoid division by zero for horizontal lines
            x_top = int((top - c) / m)
            if left <= x_top <= right:
                intersections.append((x_top, top))
    else:
        intersections.append((x1, top))  # Vertical line case

    # Check intersection with the bottom boundary (y = bottom)
    if m is not None:
        if m != 0:  # Avoid division by zero for horizontal lines
            x_bottom = int((bottom - c) / m)
            if left <= x_bottom <= right:
                intersections.append((x_bottom, bottom))
    else:
        intersections.append((x1, bottom))  # Vertical line case

    return intersections

def point_to_line_distance_and_projection(front_center, rear_center, point):
    """
    Calculates the distance between a point and a line segment, and returns the projection of the point onto the line.

    Args:
        front_center (tuple): The coordinates of the front center of the line segment.
        rear_center (tuple): The coordinates of the rear center of the line segment.
        point (tuple): The coordinates of the point.

    Returns:
        tuple: A tuple containing the distance between the point and the line segment, the slope of the line segment,
               the y-intercept of the line segment, and the coordinates of the projection of the point onto the line.

    """
    x1, y1 = front_center
    x2, y2 = rear_center
    x0, y0 = point

    # Calculate the slope and y-intercept of the line segment
    if x1 == x2:  # Vertical line
        slope = float('inf')
        intercept = None
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

    # Calculate the distance between the point and the line
    if slope == float('inf'):
        distance = abs(x0 - x1)
        projection_x = x1
        projection_y = y0
    else:
        distance = abs(slope * x0 - y0 + intercept) / math.sqrt(slope**2 + 1)
        projection_x = (slope * (y0 - intercept) + x0) / (slope**2 + 1)
        projection_y = (slope**2 * y0 + slope * x0 + intercept) / (slope**2 + 1)

    # Calculate the cross product of vectors AB and AP to determine the position of the point
    cross_product = (x2 - x1) * (y0 - y1) - (y2 - y1) * (x0 - x1)

    # Determine the sign of the distance based on the sign of the cross product
    if cross_product > 0:
        return -distance, slope, intercept, (projection_x, projection_y)  # Left side, distance is negative
    elif cross_product < 0:
        return distance, slope, intercept, (projection_x, projection_y)   # Right side, distance is positive
    else:
        return 0, slope, intercept, (projection_x, projection_y)  # Point is on the line

def process_frame(frame, frame_index):
    print(f"\nProcessing frame {frame_index}...")
    frame = undistort_image(frame)
    assert frame is not None, "Error: Failed to undistort image."

    frame_height, frame_width = frame.shape[:2]

    # Define search areas: front and rear of the car: (Xmin, Ymin, Xmax, Ymax)
    front_search_area = (
        450,
        0,
        frame_width - 450,
        max(car_corners[0][1], car_corners[1][1]),
    )
    rear_search_area = (
        450,
        min(car_corners[2][0], car_corners[3][0]),
        frame_width - 450,
        frame_height,
    )

    # Mark search areas on the original frame
    cv2.rectangle(frame, (front_search_area[0], front_search_area[1]), (front_search_area[2], front_search_area[3]), (255, 0, 0), 2)
    cv2.rectangle(frame, (rear_search_area[0], rear_search_area[1]), (rear_search_area[2], rear_search_area[3]), (255, 0, 0), 2)

    front_white_lines, front_binary = detect_white_lines(frame, front_search_area)
    rear_white_lines, rear_binary = detect_white_lines(frame, rear_search_area)

    # print("Front white lines detected:", front_white_lines)
    # print("Rear white lines detected:", rear_white_lines)

    front_line_lengths = [round(calculate_line_length(line), 2) for line in front_white_lines]
    rear_line_lengths = [round(calculate_line_length(line), 2) for line in rear_white_lines]

    front_longest_line = front_white_lines[np.argmax(front_line_lengths)] if front_white_lines else None
    rear_longest_line = rear_white_lines[np.argmax(rear_line_lengths)] if rear_white_lines else None

    cv2.line(frame, (front_longest_line[0], front_longest_line[1]), (front_longest_line[2], front_longest_line[3]), (0, 50, 255), 5)
    cv2.line(frame, (rear_longest_line[0], rear_longest_line[1]), (rear_longest_line[2], rear_longest_line[3]), (0, 50, 255), 5)

    front_intersections_raw = extend_line_to_border(front_longest_line, car_corners)
    # print(f"{front_intersections_raw=}")
    front_intersection_upper = front_intersections_raw[0] if front_intersections_raw[0][1] < front_intersections_raw[1][1] else front_intersections_raw[1]
    rear_intersections_raw = extend_line_to_border(rear_longest_line, car_corners)
    # print(f"{rear_intersections_raw=}")
    rear_intersection_lower = rear_intersections_raw[0] if rear_intersections_raw[0][1] > rear_intersections_raw[1][1] else rear_intersections_raw[1]

    cv2.circle(frame, front_intersection_upper, 10, (0, 0, 200), 5)
    cv2.circle(frame, rear_intersection_lower, 10, (0, 0, 200), 5)

    if len(front_intersection_upper) >= 2 and len(rear_intersection_lower) >= 2:
        front_intersection = front_intersection_upper
        rear_intersection = rear_intersection_lower

        distance, slope, intercept, projection = point_to_line_distance_and_projection(front_intersection, rear_intersection, car_front_center)
        offset = distance * pixel_to_mm_ratio

        print(f"Robot Offset: {offset:.2f}")

        if mode == "video":
            results.append((frame_index, round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 3), round(offset, 2)))
        else: # folder and image
            results.append((frame_index, 0, round(offset, 2)))

        if display_realtime:
            # Draw lane center line
            cv2.line(frame, (int(front_intersection[0]), int(front_intersection[1])), (int(rear_intersection[0]), int(rear_intersection[1])), (0, 250, 250), 4)

            # Mark car front center point and car corners
            cv2.circle(frame, car_front_center, 10, (20, 250, 0), -1)
            for corner in car_corners:
                cv2.circle(frame, corner, 10, (0, 0, 255), 5)

            # Mark equation of the lane center line
            cv2.putText(frame, f"y = {slope:.2f}x + {intercept:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Draw vertical distance from car front center point to lane center line
            cv2.line(frame, (int(car_front_center[0]), int(car_front_center[1])), (int(projection[0]), int(projection[1])), (0, 0, 255), 3)
            draw_position = (int((car_front_center[0] + projection[0]) / 2), int((car_front_center[1] + projection[1]) / 2) + 70)
            cv2.putText(frame, f"Distance: {offset:.2f} mm", draw_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # save the frame image with the annotation
            save_path = Path(output_folder) / "annotated_frames" / f"frame_{frame_index}_offset_{offset:.2f}.jpg"
            print(f"Saving annotated frame to {save_path}")
            cv2.imwrite(save_path.absolute().as_posix(), frame)

    # Display multiple frames
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Processed Frame", frame)

    if front_binary is not None and debug_frame:
        cv2.imshow("Front Binary Frame", front_binary)
    if rear_binary is not None and debug_frame:
        cv2.imshow("Rear Binary Frame", rear_binary)

    if mode == "image":
        cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        pass


def get_intersection_points_by_mouse(frame_img):
    """
    Detect mouse click positions on a given frame to record front and rear intersection points.

    Parameters:
    frame (numpy.ndarray): The image frame to display and interact with

    Returns:
    tuple: Coordinates of front_intersection and rear_intersection
    """
    assert frame_img is not None, "Error: Frame is None"

    # Coordinates storage
    points = []

    def mouse_callback(event, x, y, flags, param):
        # Check for left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point {len(points)}: {x}, {y}")
            cv2.circle(frame_img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame_img, f"({x}, {y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if len(points) == 2:
                # Close the window after the second click
                cv2.destroyAllWindows()

    # Create a window and set the mouse callback function
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    # Display the image and wait for two clicks
    while True:
        text = "Click on the contact points on the front and rear of the car in sequence: 1. Front, 2. Rear of car"
        cv2.putText(frame_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw car_front_center and car_corners on the frame
        cv2.circle(frame_img, car_front_center, 10, (0, 255, 0), 5)

        for i in range(len(car_corners)):
            cv2.line(frame_img, car_corners[i], car_corners[(i+1)%len(car_corners)], (0, 0, 255), 2)
        for corner in car_corners:
            cv2.circle(frame_img, corner, 10, (0, 0, 255), 5)

        # frame_img = cv2.resize(frame_img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Image", frame_img)

        if cv2.waitKey(1) & 0xFF == 27:  # Exit if ESC is pressed
            break
        if len(points) == 2:
            break

    # Ensure the window is closed
    cv2.destroyAllWindows()

    if len(points) < 2:
        raise ValueError("Two points were not selected")

    return points[0], points[1]

def manual_annote(frame, frame_index):
    front_intersection, rear_intersection = get_intersection_points_by_mouse(frame)
    distance, slope, intercept, projection = point_to_line_distance_and_projection(front_intersection, rear_intersection, car_front_center)
    offset = distance * pixel_to_mm_ratio

    print(f"Robot Offset: {offset:.2f}")

    # Append the results to button part of the results list, and in save process will sort the results by frame_index
    results.append((frame_index, 0, round(offset, 2))) # we can't get fps and time point from manual annotation


if __name__ == "__main__":
    if mode == "video":
        video_path = Path(video_path)
        if not Path.is_file(video_path):
            raise FileNotFoundError(f"Error: Video file not found at {video_path}")
        video_name = video_path.stem

        video_path_absolute = video_path.absolute().as_posix()
        cap = cv2.VideoCapture(video_path_absolute)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = frame_rate * sampling_interval
        start_frame = (start_recognition_time["start_min"] * 60 + start_recognition_time["start_sec"]) * frame_rate + round(start_recognition_time["start_ms"] * frame_rate)
        print(f"{start_recognition_time = }")
        print(f"Start frame: {start_frame}")

        missing_frame = []
        annote_frame_output_path = Path(output_folder) / "annotated_frames"
        if annote_frame_output_path.exists():
            for item in annote_frame_output_path.iterdir():
                item.unlink()  # Remove all files in the folder
        else:
            annote_frame_output_path.mkdir(parents=True, exist_ok=True)

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index >= start_frame:
                if (frame_index - start_frame) % frame_interval == 0:
                    try:
                        process_frame(frame, frame_index)
                    except (ValueError, IndexError, TypeError) as e:
                        print(f"Skipping frame {frame_index} due to error: {e}")
                        print("No worries, keep going!")
                        missing_frame.append(frame_index)

            frame_index += 1

        cap.release()
        print("\n*** IMPORTANT ***")
        print(f"Missing frames: {missing_frame}, need to manual label them")

        missing_frame_path = Path(output_folder) / "missing_frame.txt"
        with open(missing_frame_path, "w") as f:
            for item in missing_frame:
                f.write(f"{item}\n")
        csv_file_path = os.path.join(os.getcwd(), output_folder, f"vehicle_offset_{video_name}.csv")

        # Manual annotation for missing frames
        if manual_annote_missing_frame:
            print("\nLet's start to process missing frames ...")

            # Remove the output_path folder content before starting
            shutil.rmtree(Path(output_folder) / "extracted_frames", ignore_errors=True)

            for frame_index in missing_frame:
                frame = extract_frame(video_path, frame_index, output_folder)
                if frame is not None:
                    manual_annote(frame, frame_index)

    elif mode == "image":
        image_path = Path(image_path)
        if not Path.is_file(image_path):
            raise FileNotFoundError(f"Error: Image file not found at {image_path}")
        frame = cv2.imread(image_path.absolute().as_posix())
        process_frame(frame, 0)

    elif mode == "folder":
        folder_path = Path(folder_path)
        if not Path.is_dir(folder_path):
            raise FileNotFoundError(f"Error: Folder not found at {folder_path}")
        folder_name = folder_path.split("/")[-1]

        folder_path_absolute = folder_path.absolute().as_posix()

        image_files = sorted(os.listdir(folder_path_absolute))  # Sort image files in the folder
        current_index = 0  # Current image index

        while True:
            image_file = image_files[current_index]
            folder_image_path = os.path.join(folder_path_absolute, image_file)
            frame = cv2.imread(folder_image_path)
            process_frame(frame, current_index)

            key = cv2.waitKey(0)
            if key == ord("d"):  # Press "d" key to go to the next image
                current_index = min(current_index + 1, len(image_files) - 1)
            elif key == ord("a"):  # Press "a" key to go to the previous image
                current_index = max(current_index - 1, 0)
            elif key == ord("q"):  # Press "q" key to leave the loop
                break

        csv_file_path = os.path.join(os.getcwd(), output_folder, f"vehicle_offset_{folder_name}.csv")

    else:
        print("Invalid mode selected. Please choose 'video', 'image', or 'folder'.")

    cv2.destroyAllWindows()

    if results:
        if manual_annote_missing_frame:
            # Sort the results by frame_index
            results.sort(key=lambda x: x[0])

        df = pd.DataFrame(results, columns=["Frame Index", "Timestamp", "Offset"])
        # plt.plot(df["Frame Index"], df["Offset"])
        plt.figure(figsize=(6, 15))
        plt.plot(df["Offset"], df["Frame Index"])
        plt.xlabel("Offset (mm)")
        plt.ylabel("Frame Index")
        plt.title("Vehicle Offset Over Time")
        plt.savefig(os.path.join(os.getcwd(), output_folder, f"path.png"))
        plt.show()

        if mode != "image":
            df.to_csv(csv_file_path, index=False)
    else:
        print("No results to display.")
