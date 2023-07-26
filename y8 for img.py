import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("D:\\Thesis\\Dataset\\BDD10K\\BDD10K\\yolov8env\\runs\\detect\\train5\\weights\\best.pt")

w = 103 #object width in pixel
d = 280 #distance in real between object and cam
W = 138 # real width of object


focal_length = (w*d)/W # focal lentgh
print('focal_length:', focal_length)

est_dst = ((W*focal_length)/w)/100.0
print('Estimate Distance:', est_dst)
def estimate_dist(focallength, object_width, pixel_width):
    e_meter = ((focallength*object_width)/pixel_width) #in cm
    e_cm = ((focallength * object_width) / pixel_width)/100.0 #in meter
    return e_meter


# Load the image
image_path = ".jpg"
img = cv2.imread('refimg4bike02.png')
image = cv2.resize(img, (1136, 639))
# Run YOLOv8 inference on the image
results = model.predict(image, line_width=10, device=0)
boxes= results[0].boxes

rider_box = []
for box in boxes:
    cords = box.xyxy[0].tolist()
    cls = box.cls[0].item()
    conf = box.conf[0].item()

    if cls == 2:
        print('xywh', box.xywh[0].tolist())
       #print('xywhn', box.xywhn[0].tolist())
        print('xyxy', box.xyxy[0].tolist())
       #print('xyxyn', box.xyxyn[0].tolist())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        wb = box.xywh[0].tolist()[2]
        print('wb', wb)
        w_bike = x2 - x1
        #print('width', w_bike)
        d_bike = W*focal_length/w_bike
        print('d_bike', d_bike)
        #cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),3)

        # x1 = round(x1)
        # y1 = round(y1)
        # x2 = round(x2)
        # y2 = round(y2)
        d_bike = int((d_bike)/100)
        label = "%s : %.fm : %.f%%" % (results[0].names[cls], d_bike, conf)
        x1, y1, x2, y2 = map(round, [ x1, y1, x2, y2 ])



        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 165, 0), 3)
       # cv2.putText(image, results[0].names[3], (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 3)
        #cv2.putText(image,  f'Dis:{int(d_bike)}cm', (x2-50, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 3)

        cv2.putText(image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 3)
        #cv2.imwrite(r'C:\Users\abm_0\OneDrive\Desktop\ref_img_01', image)


    else:
        continue


#cv2.imwrite(r'C:\Users\abm_0\OneDrive\Desktop\ref_img_01', image)
cv2.imwrite(r'C:\Users\abm_0\OneDrive\Desktop\Ref Image07 bike01.jpg', image)

#print(len(results[0]))
print(results[0].names)
#boxes= results[0].boxes
#box = boxes[0]
print(box.xyxy)
print(box.xywh)
#x1, y1, x2, y2 = box.xyxy
#w = x2 - x1
#h = y2 - y1
#print('w:', w)
#print('h:', h)
# Visualize the results on the image
annotated_image = results[0].plot()

# Display the annotated image
#cv2.imshow("YOLOv8 Inference", annotated_image)
#cv2.imwrite(r'C:\Users\abm_0\OneDrive\Desktop\Ref Image.jpg', annotated_image)
