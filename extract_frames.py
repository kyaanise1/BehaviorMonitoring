import cv2
import os

# Input video
video_path = r"C:\Acads\4th Year\1st Semester\Thesis\Behavior Monitoring\video\Day_01_2026-08-27_0600_to_0700.mkv"

# Folder where extracted images will be saved
output_folder = "extracted_frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    print("Error: Could not read FPS from the video.")
    cap.release()
    exit()

duration = total_frames / fps

print("FPS:", fps)
print("Total frames:", total_frames)
print("Duration:", duration / 3600, "hours")

# Extract one frame every 5 seconds
interval = 5

# Day number
day_number = 1  # Change this to the appropriate day number

# Starting frame number for this day
start_number = day_number * 1000

current_time = 0
image_count = 0

while current_time < duration:

    # Move video to the desired time
    cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)

    # Read the frame
    ret, frame = cap.read()

    if not ret:
        break

    # Calculate filename number
    frame_number = start_number + image_count

    # Save the frame
    filename = os.path.join(
        output_folder,
        f"frame_{frame_number:05d}.jpg"
    )

    cv2.imwrite(filename, frame)

    image_count += 1

    print(f"Saved: {filename} | Time: {current_time:.1f} seconds")

    # Move forward 5 seconds
    current_time += interval

# Release video
cap.release()

print("\nExtraction complete!")
print("Total images:", image_count)
print(f"Frame numbering: frame_{start_number:05d}.jpg onwards")


#pip install labelImg
#labelimg