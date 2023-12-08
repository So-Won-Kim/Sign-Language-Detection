import sys
import os
import cv2
import mediapipe as mp

def is_enough_motion(frame, previous_frame, motion_threshold=1000):
    # Calculate absolute difference between current and previous frame
    diff = cv2.absdiff(frame, previous_frame)
    motion_score = cv2.sumElems(diff)[0]
    return motion_score > motion_threshold

def calculate_average_bitrate(video_path):
    cap = cv2.VideoCapture(video_path)
    total_bitrate = 0
    frame_count = 0

    while True:
        ret, _ = cap.read()
        if not ret:
            break

        total_bitrate += int(cap.get(cv2.CAP_PROP_BITRATE))
        frame_count += 1

    cap.release()

    if frame_count == 0:
        return 0  # Avoid division by zero
    return total_bitrate / frame_count

def capture_frames(input_video_path, output_directory, motion_threshold):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    average_bitrate = calculate_average_bitrate(input_video_path)

    frame_number = 0
    previous_frame = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with mediapipe hands
        results = hands.process(rgb_frame)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            # Zoom and focus on the detected hand
            hand_landmarks = results.multi_hand_landmarks[0]  # Assuming only one hand is detected
            bounding_box = calculate_hand_bounding_box(hand_landmarks, width, height)

            # Crop the frame to the bounding box
            cropped_frame = frame[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]].copy()

            # Save the cropped frame as an image
            image_filename = os.path.join(output_directory, f"captured_frame_{frame_number}.jpg")
            cv2.imwrite(image_filename, cropped_frame)

            frame_number += 1

        # Update the previous frame
        previous_frame = frame.copy()

    print(f"Frames captured: {frame_number}")

    # Release the VideoCapture object
    cap.release()
    hands.close()

def calculate_hand_bounding_box(hand_landmarks, frame_width, frame_height):
    min_x = min(int(l.x * frame_width) for l in hand_landmarks.landmark)
    min_y = min(int(l.y * frame_height) for l in hand_landmarks.landmark)
    max_x = max(int(l.x * frame_width) for l in hand_landmarks.landmark)
    max_y = max(int(l.y * frame_height) for l in hand_landmarks.landmark)

    # Expand the bounding box to ensure the entire hand is captured
    bounding_box = (max(0, min_x - 20), max(0, min_y - 20), min(frame_width, max_x + 20), min(frame_height, max_y + 20))

    return bounding_box

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python FILE.py <input_video.mp4> <output_directory>")
        sys.exit(1)

    input_video_path = sys.argv[1]
    output_directory = sys.argv[2]
    motion_threshold = 1000  # Set your desired motion threshold here

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Initialize mediapipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    capture_frames(input_video_path, output_directory, motion_threshold)

    # Close mediapipe hands
    hands.close()
