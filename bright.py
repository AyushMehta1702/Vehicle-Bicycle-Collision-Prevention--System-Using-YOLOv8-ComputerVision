import cv2

def adjust_brightness_contrast(frame, brightness=0, contrast=0):
    # Apply brightness and contrast adjustment
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast/127.0, beta=brightness)

    return adjusted_frame

# Open video file
video = cv2.VideoCapture('test13.mp4')

# Check if video file opened successfully
if not video.isOpened():
    print("Error opening video file.")

# Read the first frame
ret, frame = video.read()

# Specify desired brightness and contrast values
brightness = 2
contrast =  150

# Process video frames
while ret:
    # Adjust brightness and contrast of the frame
    adjusted_frame = adjust_brightness_contrast(frame, brightness, contrast)

    # Display the adjusted frame
    cv2.imshow('Adjusted Frame', adjusted_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read the next frame
    ret, frame = video.read()

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()
