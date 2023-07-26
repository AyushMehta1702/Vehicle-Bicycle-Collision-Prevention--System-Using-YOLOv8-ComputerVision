import cv2
import torch
import numpy as np
#from main import perspective_transform
import time


model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\\YOLOv5\\runs\\train\\exp9\\weights\\best.pt')
model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()


classes = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'rider', 'truck']

def score_frame(frame):
    results = model([frame])
    labels, cord = results.xyxyn[0][:, -1].detach().cpu().numpy(), results.xyxyn[0][:, :-1].detach().cpu().numpy()
    return labels, cord

def class_to_label(x):
    return classes[int(x)]

def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.6: # Confidence threshold
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            label = f"{class_to_label(labels[i])}: {row[4]*100:.2f}%"
            if class_to_label(labels[ i ]) == 'bicycle':

              #  bbox_area = (x2 - x1) * (y2 - y1)
                bbox_height = y2 - y1

               # distance = 1 / bbox_area
                distance = 1000 / bbox_height
                label += f", Distance: {distance:.2f}"
               # label1 += f", Distance1: {distance1:.2f}m"
               # print(bbox_area)
                print(bbox_height)

            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return frame


input_vid = cv2.VideoCapture('test13.mp4')
#input_vid = cv2.resize(input_vid, (1136,639))
output_path = 'output_video_04.avi'

# cap = cv2.VideoCapture(input1)

#iv = cv2.resize(input_vid, (1136, 639))
fps = input_vid.get(cv2.CAP_PROP_FPS)
frame_width = int(input_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

codec = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video013.mp4', codec, fps, (frame_width, frame_height))
#out = cv2.VideoWriter('outp.mp4',cv2.VideoWriter_fourcc(*'MP4v'), 20, (1139,639))
while True:
    success, input_img = input_vid.read()
    # frame = cv2.resize(img, (640,480))

    if input_img is None:
        print('No Frame')
        continue

    #out_img = detect_lanes(input_img)
    start_time = time.perf_counter()
    results = score_frame(input_img)
    frame = plot_boxes(results, input_img)
    end_time = time.perf_counter()
    fpss = 1 / np.round(end_time - start_time, 3)
    cv2.putText(frame, f'FPS: {int(fpss)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)


    out.write(frame)
    #frame_width = int(out_img.shape[0])
   # frame_height = int(out_img.shape[1])

    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   # out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))
    # frame = cv2.resize(out_img, (640,480))
    cv2.imshow('Original', frame)
    # cv2.imshow('Warped', out_img)

    cv2.waitKey(1)
    #if cv2.waitKey(1) or 0xFF == ord('q'):
        #break

#input_vid.release()
out.release()

# Closes all the frames
#cv2.destroyAllWindows()
