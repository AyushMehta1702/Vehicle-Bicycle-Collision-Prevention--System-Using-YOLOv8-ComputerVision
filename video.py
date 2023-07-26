import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the image

# frame
#img = cv2.imread("./images/Frame0.jpg")
img = cv2.imread("123.jpg")

h_img, w_img, _ = img.shape
# resizing parameter
h_ori = int(h_img * 1)
w_ori = int(w_img * 1)
# Binary thresholding
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (w_ori, h_ori))
gray = cv2.resize(gray, (w_ori, h_ori))
blur = cv2.GaussianBlur(gray, (3, 3), 2)
binary_img = cv2.threshold(blur, 150, 255, type=cv2.THRESH_BINARY)[1]
#cv2.imwrite("binary.jpg", binary_img)
cv2.imshow("sasd", binary_img)

print("www")
# perspective transformation

tl = (570, 500)
bl = (360, 680)
tr = (820, 500)
br = (1200, 680)
pts1 = np.float32([tl, bl, tr, br])
pts2 = np.float32([[0, 0], [0, h_ori], [w_ori, 0], [w_ori, h_ori]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
matrix_inv = cv2.getPerspectiveTransform(pts2, pts1)


points = np.array([tl, tr, br, bl])

transformed_frame = cv2.warpPerspective(binary_img, matrix, (w_ori, h_ori), flags=cv2.INTER_LINEAR)

# code for sliding window
#print(transformed_frame.shape)
#plt.imshow(transformed_frame)
#cv2.imshow("trans", transformed_frame)


histogram = np.sum(transformed_frame[600:, :], axis=0)
#arrr = np.linspace(0, w_ori, w_ori)
#plt.plot(arrr,histogram)
#plt.show()
print(histogram.shape)

midpoint = 600
l_base = np.argmax(histogram[0:550])
r_base = np.argmax(histogram[750:1000]) + 750
rr_base = np.argmax(histogram[1010:w_ori]) + 1010

#print(l_base,r_base,rr_base)
#print("xxxxxxx")

win_width = 50
win_height = 30
y = h_ori
r = 0
rl = []
rrl = []
ll = []
i = 1
previous_left_area = 0
previous_right_area = 0
previous_rright_area = 0

if l_base != 0 and r_base != 0:
    i= 0
    while y > 0:
        #code for left lane
        left_window = transformed_frame[y - 30:y, l_base - win_width:l_base + win_width]
        cons, _ = cv2.findContours(left_window, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        if cons:
            largest_contour = max(cons, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                new_x = int(M["m10"] / M["m00"])
                new_y = int(M["m01"] / M["m00"])
                current_left_area = M["m00"]
                if current_left_area < (previous_left_area // 2):
                    pass
                else:
                    l_base = l_base - win_width + new_x
                    ll.append((l_base, y - 30 + new_y))
                    previous_left_area = current_left_area
        cv2.rectangle(transformed_frame, (l_base - win_width, y), (l_base + win_width, y - 30), (255, 255, 255), 1)
        i = i+1
        # code for right lane
        right_window = transformed_frame[y - 30:y, r_base - win_width:r_base + win_width]
        cons, _ = cv2.findContours(right_window, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        if cons:
            largest_contour = max(cons, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:

                new_x = int(M["m10"] / M["m00"])
                new_y = int(M["m01"] / M["m00"])
                current_right_area = M["m00"]
                if current_right_area < (previous_right_area // 2):
                    pass
                else:
                    r_base = r_base - win_width + new_x
                    rl.append((r_base, y - 30 + new_y))
                    previous_right_area = current_right_area
        cv2.rectangle(transformed_frame, (r_base - win_width, y), (r_base + win_width, y - 30), (255, 255, 255), 1)

        if rr_base != 0:
            # code for next right lane
            next_right_window = transformed_frame[y - 30:y, rr_base - win_width:rr_base + win_width]
            cons, _ = cv2.findContours(next_right_window, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            if cons:
                largest_contour = max(cons, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:

                    new_x = int(M["m10"] / M["m00"])
                    new_y = int(M["m01"] / M["m00"])
                    current_rright_area = M["m00"]
                    if current_rright_area < (previous_rright_area // 2):
                        pass
                    else:
                        rr_base = rr_base - win_width + new_x
                        rrl.append((rr_base, y - 30 + new_y))
                        previous_rright_area = current_rright_area
            cv2.rectangle(transformed_frame, (rr_base - win_width, y), (rr_base + win_width, y - 30), (255, 255, 255), 1)

        y -= 32
    #print(f"the windiws ae: {i}")
    #cv2.imshow("window",transformed_frame)
lxx = [t[0] for t in ll]
lxy = [t[1] for t in ll]
rxx = [t[0] for t in rl]
rxy = [t[1] for t in rl]
rrxx = [t[0] for t in rrl]
rrxy = [t[1] for t in rrl]

#print(f"left lane: {len(lxx)}, right lane: {len(rxx)}, rr lane: {len(rrxx)}")
out_img = np.zeros((h_ori, w_ori, 3), dtype=np.uint8)
line_width = 15
green_color = [0, 255, 0] # BGR
red_color = [0, 0, 150]
if len(ll) > 6:
    l_lane = np.polyfit(lxy, lxx, 2)
    min_ = min(lxy)
    max_ = max(lxy)
    l_y_points = np.linspace(min_, max_, 30, dtype=int)
    l_x_points = np.array(l_lane[0] * l_y_points ** 2 + l_lane[1] * l_y_points + l_lane[2], dtype=int)
    if len(ll) > 24:
        for index in range(len(l_x_points) - 1):
            cv2.line(out_img, (l_x_points[index], l_y_points[index]), (l_x_points[index + 1],
                                                                     l_y_points[index + 1]), red_color, line_width)
    else:
        for index in range(len(l_x_points) - 1):
            cv2.line(out_img, (l_x_points[index], l_y_points[index]), (l_x_points[index + 1],
                                                                     l_y_points[index + 1]), green_color, line_width)
if len(rl) > 6:
    r_lane = np.polyfit(rxy, rxx, 2)
    min_ = min(rxy)
    max_ = max(rxy)
    r_y_points = np.linspace(min_, max_, 30, dtype=int)
    r_x_points = np.array(r_lane[0] * r_y_points ** 2 + r_lane[1] * r_y_points + r_lane[2], dtype=int)
    if len(rl) > 24:
        for index in range(len(r_x_points) - 1):
            cv2.line(out_img, (r_x_points[index], r_y_points[index]), (r_x_points[index + 1],
                                                                     r_y_points[index + 1]), red_color, line_width)
    else:
        for index in range(len(r_x_points) - 1):
            cv2.line(out_img, (r_x_points[index], r_y_points[index]), (r_x_points[index + 1],
                                                                     r_y_points[index + 1]), green_color, line_width)
if len(rrl) > 6:
    rr_lane = np.polyfit(rrxy, rrxx, 2)
    min_ = min(rrxy)
    max_ = max(rrxy)
    rr_y_points = np.linspace(min_, max_, 30, dtype=int)
    rr_x_points = np.array(rr_lane[0] * rr_y_points ** 2 + rr_lane[1] * rr_y_points + rr_lane[2], dtype=int)
    if len(rrl) > 24:
        for index in range(len(rr_x_points) - 1):
            cv2.line(out_img, (rr_x_points[index], rr_y_points[index]), (rr_x_points[index + 1],
                                                                     rr_y_points[index + 1]), red_color, line_width)
    else:
        for index in range(len(rr_x_points) - 1):
            cv2.line(out_img, (rr_x_points[index], rr_y_points[index]), (rr_x_points[index + 1],
                                                                     rr_y_points[index + 1]), green_color, line_width)


if len(rl) <= 24 and len(rrl) > 5:
    #print(r_y_points)
    points_right = []
    points_rr = []
    for x, y in zip(r_x_points, r_y_points):
        points_right.append((x, y))
    for x, y in zip(rr_x_points, rr_y_points):
        points_rr.append((x, y))
    reversed_list = list(reversed(points_rr))
    listu = np.array(points_right + reversed_list, dtype=np.int32)
    cv2.fillPoly(out_img, [listu], color=(0, 255, 0))
else:
    if len(rl) > 18:
        points_right = []
        points_rr = []
        for x, y in zip(r_x_points, r_y_points):
            points_right.append((x, y))
        for x, y in zip(rr_x_points, rr_y_points):
            points_rr.append((x, y))
        reversed_list = list(reversed(points_rr))
        listu = np.array(points_right + reversed_list, dtype=np.int32)
        cv2.fillPoly(out_img, [listu], color=(0, 0, 255))


transformedback_frame = cv2.warpPerspective(out_img, matrix_inv, (w_ori, h_ori), flags=cv2.INTER_LINEAR)
Final_img = cv2.addWeighted(img, 0.9, transformedback_frame, 1, 0)
#cv2.imwrite("outputred.png", Final_img)
cv2.imshow("asdad", Final_img)

cv2.imshow("ad", transformed_frame)

cv2.waitKey(5000)
cv2.destroyAllWindows()

