import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

import glob
from scipy.signal import find_peaks
import time
from yolov5 import *

def adjust_brightness_contrast(frame, brightness=10, contrast=150):
    # Apply brightness and contrast adjustment
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast/127.0, beta=brightness)
    return adjusted_frame

def resize(img):
    return cv2.resize(img, (1136,639))

def thresholding_img(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_channel = hls[ :, :, 0 ]
    l_channel = hls[ :, :, 1 ]
    s_channel = hls[ :, :, 2 ]

    _, sxbinary = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)
    sxbinary = cv2.GaussianBlur(sxbinary, (3, 3), 0)

    # 2. Sobel edge detection on the L channel
    # l_channel = hls[:, :, 1]
    sobelx = cv2.Sobel(sxbinary, cv2.CV_64F, 1, 0, 3)
    sobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(sxbinary, cv2.CV_64F, 0, 1, 3)
    sobely = np.absolute(sobely)
    sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[ (scaled_sobel >= 120) & (scaled_sobel <= 255) ] = 1

    # 3. Threshold on the S channel
    s_channel = hls[ :, :, 2 ]
    _, s_binary = cv2.threshold(s_channel, 130, 255, cv2.THRESH_BINARY)
    # s_binary = np.zeros_like(s_channel)
    # s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1

    # 4. Threshold on the R channel
    r_channel = img[ :, :, 1 ]
    _, r_thresh = cv2.threshold(r_channel, 120, 255, cv2.THRESH_BINARY)
    # r_binary = np.zeros_like(r_channel)
    # r_binary[(r_channel >= 200) & (r_channel <= 255)] = 1
    rs_binary = cv2.bitwise_or(s_binary, r_thresh)

    combined_binary = cv2.bitwise_or(rs_binary, sxbinary.astype(np.uint8))

    return combined_binary


# Define source (src) and destination (dst) points


def perspective_transform(img):

    src = np.float32([
        [ 380, 450 ],
        [ 750, 450 ],
        [ 100, 639 ],
        [ 1000, 639 ] ])

    dst = np.float32([
        [ 0, 0 ],
        [ img.shape[ 1 ], 0 ],
        [ 0, img.shape[ 0 ] ],
        [ img.shape[ 1 ], img.shape[ 0 ] ] ])

    img_size = (img.shape[ 1 ], img.shape[ 0 ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped_img, M, Minv

# USAGE

# warped_image, M, Minv = perspective_transform(thresh_img, src, dst)

# plt.figure()
# plt.imshow(warped_image)

def sliding_window_search(binary_warped):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    peaks, _ = find_peaks(histogram, distance= 300, height=100 )
    peak_bases = sorted(peaks)

    margin = 40
    minpix = 100
    nwindows = 20
    window_height = int(binary_warped.shape[0] // nwindows)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    lane_inds_list = []

    for base in peak_bases:
        x_current = base
        lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            lane_inds.append(good_inds)

            if len(good_inds) > minpix:
                x_current = int(np.mean(nonzerox[good_inds]))



        lane_inds = np.concatenate(lane_inds)
        lane_inds_list.append(lane_inds)

    lane_type_list = []
    gap_threshold = 10

    for lane_inds in lane_inds_list:
        if len(lane_inds) <= 1:
            avg_gap = 0
        else:
            sorted_inds = sorted(lane_inds)
            gaps = [sorted_inds[i + 1] - sorted_inds[i] for i in range(len(sorted_inds) - 1)]
            avg_gap = np.mean(gaps)
            print(avg_gap)

        if avg_gap > gap_threshold:
            lane_type = 'vehicle'
            print(lane_type)
        elif 3 < avg_gap <= gap_threshold:
            lane_type = 'bicycle'
            print(lane_type)
        else:
            lane_type = 'Solid'
            print(lane_type)
        lane_type_list.append(lane_type)

    return lane_inds_list, nonzerox, nonzeroy, lane_type_list, out_img

def overlay_lanes(binary_warped, Minv, input_img, lane_inds_list, nonzerox, nonzeroy, lane_type_list):
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))*255

    for i, (lane_inds, lane_type) in enumerate(zip(lane_inds_list, lane_type_list)):
        color = (0, 255, 255) if lane_type == 'vehicle' else (255, 0, 0) if lane_type == 'bicycle' else (255, 255, 255)
        color_warp[nonzeroy[lane_inds], nonzerox[lane_inds]] = color

  #  color_warp[nonzeroy[lane_inds_list], nonzerox[lane_inds_list]] = [255, 0, 0]
    newwarp = cv2.warpPerspective(color_warp, Minv, (input_img.shape[1], input_img.shape[0]))
    result = cv2.addWeighted(input_img, 0.5, newwarp, 1, 0)
    return result

def detect_lanes(input_img):
    img1 = resize(input_img)
    #resized = input_img
    resized = adjust_brightness_contrast(img1)
    thresh_img = thresholding_img(resized)
    cv2.imshow("qqq", thresh_img)
    warped_img, M, Minv = perspective_transform(thresh_img)
    lane_inds_list, nonzerox, nonzeroy, lane_type_list, swindow = sliding_window_search(warped_img)
    #left_curverad, right_curverad, vehicle_position = calculate_curvature(warped_img, lane_inds_list, nonzerox, nonzeroy)
    result = overlay_lanes(warped_img, Minv, resized, lane_inds_list, nonzerox, nonzeroy, lane_type_list)
    return result, swindow


input_vid = cv2.VideoCapture('test04.mp4')
#input_vid = cv2.resize(input_vid, (1136,639))
output_path = 'output_video_04.avi'

# cap = cv2.VideoCapture(input1)

#iv = cv2.resize(input_vid, (1136, 639))
fps = input_vid.get(cv2.CAP_PROP_FPS)
frame_width = int(input_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

codec = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('outvid01.mp4', codec, fps, (frame_width, frame_height))
#out = cv2.VideoWriter('outp.mp4',cv2.VideoWriter_fourcc(*'MP4v'), 20, (1139,639))
while True:
    success, input_img = input_vid.read()
    # frame = cv2.resize(img, (640,480))

    #if input_img is None:
    #    print('No Frame')
       # continue

    out_img, swindow = detect_lanes(input_img)
    start_time = time.perf_counter()
    results = score_frame(out_img)
    frame = plot_boxes(results, out_img)
    end_time = time.perf_counter()
    fpss = 1 / np.round(end_time - start_time, 3)
    cv2.putText(frame, f'FPS: {int(fpss)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)


    #out.write(frame)
    #frame_width = int(out_img.shape[0])
   # frame_height = int(out_img.shape[1])

    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   # out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))
    # frame = cv2.resize(out_img, (640,480))
    cv2.imshow('Original', frame)
    cv2.imshow('sliding win:', swindow)
    # cv2.imshow('Warped', out_img)

    cv2.waitKey(1)
    #if cv2.waitKey(1) or 0xFF == ord('q'):
      #  break

input_vid.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()

