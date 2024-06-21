import cv2
import shutil
from pathlib import Path

def extract_frame(video_path, frame_number, output_path):
    """
    Extract a specific frame from a video, save it to the output directory, and return the frame image.

    Parameters:
    video_path (str or Path): Path to the video file
    frame_number (int): Frame number to extract
    output_path (str or Path): Directory to save the extracted frame

    Returns:
    numpy.ndarray: The extracted frame image
    """
    # Ensure video_path and output_path are Path objects
    video_path = Path(video_path)
    output_path = Path(output_path) / "extracted_frames"
    assert video_path.is_file(), f"Error: {video_path} does not exist"

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Set the video frame position to the specified frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the specified frame
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        cap.release()
        return None
    else:
        print(f"\nFrame {frame_number} extracted successfully")

    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the frame image
    frame_path = output_path / f"frame_{frame_number}.jpg"
    cv2.imwrite(str(frame_path), frame)

    # Release the video capture object
    cap.release()

    # Return the frame image
    return frame


if __name__ == "__main__":
    video_path = "../Data/case2_hdbscan_test2222/case2_hdbscan_test2.mp4"
    frame_numbers = [10, 50, 100, 150]

    output_path = "../Data/case2_hdbscan_test2222/"

    extracted_frames = []

    # Loop to extract frames
    for frame_number in frame_numbers:
        frame = extract_frame(video_path, frame_number, output_path)
        if frame is not None:
            extracted_frames.append(frame)