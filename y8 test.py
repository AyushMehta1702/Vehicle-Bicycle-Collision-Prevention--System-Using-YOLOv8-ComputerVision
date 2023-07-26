import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("D:\\Thesis\\Dataset\\BDD10K\\BDD10K\\yolov8env\\runs\\detect\\train5\\weights\\best.pt")

# Open the video file
video_path = "test04.mp4"
##cv2.resize(img, (1136,639))
cap = cv2.VideoCapture(video_path)

output_path = "D:\\Thesis\\Dataset\\BDD10K\\BDD10K\\yolov8env\\Out_Video\\out_vid_02.mp4"

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

codec = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, codec, 30, (1136, 639))
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame = cv2.resize(frame, (1136, 639))
        # Run YOLOv8 inference on the frame
        results = model.predict(frame, line_width=10, device=0)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1)== ord('q'):
            break

cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()