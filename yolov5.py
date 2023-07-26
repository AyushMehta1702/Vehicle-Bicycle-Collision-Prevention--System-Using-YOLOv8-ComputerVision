import cv2
import torch
import numpy as np
#from main import perspective_transform
import time


model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\\YOLOv5\\runs\\train\\exp9\\weights\\last.pt')
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
