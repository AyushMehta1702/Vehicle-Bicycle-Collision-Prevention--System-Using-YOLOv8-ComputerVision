import cv2
from ultralytics import YOLO


model = YOLO("D:\\Thesis\\Dataset\\BDD10K\\BDD10K\\yolov8env\\runs\\detect\\train5\\weights\\best.pt")

focal_length2 = ((103*280)/138)
focal_length1 = ((43*2000)/175)

def process_boxes(image, focal_length1, focal_length2):

    results = model.predict(image, line_width=10, device=0)
    boxes = results[ 0 ].boxes

    color = [(255,165,255),(0,69,255),(0,165, 255),(128,0,128),(0,69,255)]

    for box in boxes:
        cords = box.xyxy[ 0 ].tolist()
        cls = box.cls[ 0 ].item()
        conf = box.conf[ 0 ].item()

        # Use the correct focal length for each class
        if cls in [0,2]:  # Replace with your actual class IDs
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w_obj = x2 - x1

            if cls == 0:
                W = 175
                focal_length = focal_length1
            elif cls == 2:
                W = 138
                focal_length = focal_length2

            d_obj = ((W*focal_length)/w_obj)

            x1, y1, x2, y2 = map(round, [ x1, y1, x2, y2 ])
            # print('cls:', cls)
            # print('FL:', focal_length)
            # print('w_px:', w_obj)
            # print('W:', W)
            d_obj = (d_obj / 100)
            print('dist:', d_obj)

            # print('-----------^')

            label = "%s:%.fm : %.f%%" % (results[0].names[cls], d_obj, conf)
            label01 = "%.fm" % (d_obj)

            cv2.rectangle(image, (x1, y1), (x2, y2), color[int(cls)], 1)
            cv2.putText(image, label01, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color[int(cls)], 1)

    return image

# img = cv2.imread('refimg4bike02.png')
# image = cv2.resize(img, (1136, 639))
#
# outimg = process_boxes(image, focal_length1, focal_length2)
# cv2.imwrite(r'C:\Users\abm_0\OneDrive\Desktop\RefEstImg01.jpg', outimg)

video_path = "F_01.mp4"
##cv2.resize(img, (1136,639))
cap = cv2.VideoCapture(video_path)

#----------OUT VID----------
output_path = "C:\\Users\\abm_0\\OneDrive\\Desktop\\out_vid_F01.mp4"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, codec, 30, (1136, 639))


while cap.isOpened():

    success, frame = cap.read()

    if success:
        frame = cv2.resize(frame, (1136, 639))

        outimg = process_boxes(frame, focal_length1, focal_length2)

        #results = model.predict(frame, show_labels=True)

        ############


        ###########


        #annotated_frame = results[0].plot()


        cv2.imshow("YOLOv8 Inference", outimg)
        out.write(outimg)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()