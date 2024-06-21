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
from evaluation_lane_centering import manual_annote


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    if '_BASE_' in config:
        base_path = config.pop('_BASE_')
        base_config = load_config(os.path.join(os.path.dirname(config_path), base_path))
        base_config.update(config)
        return base_config
    return config

# Load configuration
print("xxxx")

config = load_config("evaluation_one_frame_config.yaml")

# Accessing the parameters from the config
mode = config.get('mode', 'video')
video_path = config.get('video_path', 'Data/exp_lab2.mp4')
calib_file_path = config.get('calib_file_path', 'calibration_images/exp_lab2/camera_calibration.npz')
output_folder = config.get('output_folder', f'Data/robot_offset_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
car_corners = config.get('car_corners', [(790, 295), (1151, 307), (763, 771), (1151, 764)])
car_front_center = config.get('car_front_center', (966, 304))
mm_to_pixel_ratio = config.get('mm_to_pixel_ratio', 1.416)
pixel_to_mm_ratio = 1 / mm_to_pixel_ratio
display_realtime = config.get('display_realtime', True)
use_calibration = config.get('use_calibration', False)
debug_frame = config.get('debug_frame', False)
frame_index = config.get('frame_index', [])


for frame_index in frame_index:
    frame = extract_frame(video_path, frame_index, output_folder)
    if frame is not None:
        manual_annote(frame, frame_index)