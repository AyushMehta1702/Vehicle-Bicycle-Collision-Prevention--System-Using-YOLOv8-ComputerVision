import cv2
i = 0
cap = cv2.VideoCapture('test13.mp4')
while True:
    #i = 0
    ret, img = cap.read()
    cv2.imshow("video stream", img)
    cv2.imwrite(f"./images/Frame{i}.jpg",img)
    i = i+1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()