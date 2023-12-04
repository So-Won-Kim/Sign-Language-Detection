import cv2
import sys
import os

def get_average_bitrate(video_path):
    cap = cv2.VideoCapture(video_path)
    average_bitrate = int(cap.get(cv2.CAP_PROP_BITRATE))
    cap.release()
    return average_bitrate

def capture_frames_with_motion(video_path, motion_threshold):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    script_directory = os.path.dirname(os.path.abspath(__file__))
    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, motion_threshold, 255, cv2.THRESH_BINARY)
        motion_pixel_count = cv2.countNonZero(thresh)

        if motion_pixel_count > motion_threshold:
            frame_count += 1
            output_path = os.path.join(script_directory, f"motion_frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Frame {frame_count} with motion captured and saved.")

        prev_gray = gray

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file.py /path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    motion_threshold = 1000  # Adjust this value as needed

    if not os.path.isfile(video_path):
        print("Error: Video file does not exist.")
        sys.exit(1)

    average_bitrate = calculate_average_bitrate(video_path)
    print(f"Average Bitrate: {average_bitrate} bps")

    capture_frames_with_motion(video_path, motion_threshold)
