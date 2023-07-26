
def perspective_transform(img):

    src = np.float32([
        [ 380, 450 ],
        [ 750, 450 ],
        [ 50, 639 ],
        [ 1136, 639 ] ])

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


def perspective_transform_point(M, x, y):

    point = np.array([ x, y, 1 ])
    transformed_point = np.dot(M, point)
    return transformed_point[ 0 ] / transformed_point[ 2 ], transformed_point[ 1 ] / transformed_point[ 2 ]


def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[ 1 ], frame.shape[ 0 ]
    _, M, _ = perspective_transform(frame)

    for i in range(n):
        row = cord[ i ]
        if row[ 4 ] >= 0.6:  # Confidence threshold
            x1, y1, x2, y2 = int(row[ 0 ] * x_shape), int(row[ 1 ] * y_shape), int(row[ 2 ] * x_shape), int(
                row[ 3 ] * y_shape)

            x1_transformed, y1_transformed = perspective_transform_point(M, x1, y1)
            x2_transformed, y2_transformed = perspective_transform_point(M, x2, y2)
            if class_to_label(labels[ i ]) == 'bicycle':
                # Check if bicycle is in lane
                if is_in_lane(x1_transformed, y1_transformed, x2_transformed, y2_transformed):
                    bgr = (0, 255, 0)  # Green box: bicycle is safe
                else:
                    bgr = (0, 0, 255)  # Red box: bicycle is not safe
            else:
                bgr = (255, 255, 255)  # White box for all other classes

            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            label = f"{class_to_label(labels[ i ])}: {row[ 4 ] * 100:.2f}%"
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame


def is_in_lane(x1, y1, x2, y2):

    lane_left = 0
    lane_right = 1136
    lane_top = 450
    lane_bottom = 639


    return lane_left <= x1 <= lane_right and lane_left <= x2 <= lane_right and lane_top <= y1 <= lane_bottom and lane_top <= y2 <= lane_bottom
