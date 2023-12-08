import sys
import os
import cv2
import mediapipe as mp

def is_enough_motion(frame, previous_frame, motion_threshold=1000):
    # Calculate absolute difference between current and previous frame
    diff = cv2.absdiff(frame, previous_frame)
    motion_score = cv2.sumElems(diff)[0]
    return motion_score > motion_threshold

def capture_frames(input_video_path, output_directory, threshold_bitrate):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate bitrate
    bitrate = int(cap.get(cv2.CAP_PROP_BITRATE))

    print(f"Video bitrate: {bitrate} bps")

    if bitrate < threshold_bitrate:
        print("Capturing frames...")

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
                # Here you can add more logic or filters based on hand landmarks
                pass

            # Check if there is enough motion to capture the frame
            if previous_frame is None or is_enough_motion(frame, previous_frame):
                # Save the frame as an image
                image_filename = os.path.join(output_directory, f"captured_frame_{frame_number}.jpg")
                cv2.imwrite(image_filename, frame)

                frame_number += 1

            # Update the previous frame
            previous_frame = frame.copy()

        print(f"Frames captured: {frame_number}")

        # Release the VideoCapture object
        cap.release()

    else:
        print("Video has sufficient motion. No frames captured.")
    hands.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python video_capture_2.py <input_video.mp4> <output_directory>")
        sys.exit(1)

    input_video_path = sys.argv[1]
    output_directory = sys.argv[2]
    threshold_bitrate = 500000  # Set your desired threshold bitrate here

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Initialize mediapipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    capture_frames(input_video_path, output_directory, threshold_bitrate)

    # Close mediapipe hands
    hands.close()
