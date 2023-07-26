import cv2

def apply_histogram_equalization(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized_frame = cv2.equalizeHist(gray_frame)

    # Convert back to BGR color space
    equalized_frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)

    return equalized_frame

def apply_adaptive_histogram_equalization(frame, clip_limit=2.0, grid_size=(8, 8)):
    # Convert frame to LAB color space
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split the LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab_frame)

    # Apply adaptive histogram equalization to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    equalized_l_channel = clahe.apply(l_channel)

    # Merge the equalized L channel with the original A and B channels
    equalized_lab_frame = cv2.merge((equalized_l_channel, a_channel, b_channel))

    # Convert back to BGR color space
    equalized_frame = cv2.cvtColor(equalized_lab_frame, cv2.COLOR_LAB2BGR)

    return equalized_frame

# Open video file
video = cv2.VideoCapture('test13.mp4')

# Check if video file opened successfully
if not video.isOpened():
    print("Error opening video file.")

# Get video properties
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc(*'mp4v')

# Create output video writer
output_video = cv2.VideoWriter('output_video.mp4', codec, fps, (width, height))

# Process video frames
while True:
    # Read the frame
    ret, frame = video.read()

    # Check if frame was read successfully
    if not ret:
        break

    # Apply histogram equalization to the frame
    equalized_frame = apply_histogram_equalization(frame)

    # Apply adaptive histogram equalization to the frame
    # equalized_frame = apply_adaptive_histogram_equalization(frame)

    # Write the processed frame to the output video
    output_video.write(equalized_frame)

    # Display the processed frame
    cv2.imshow('Processed Frame', equalized_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and video writer
video.release()
output_video.release()

# Close all windows
cv2.destroyAllWindows()
