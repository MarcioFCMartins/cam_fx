import cv2
import numpy as np
from apply_filters import gray_scale
from apply_filters import threshhold
from apply_filters import detect_motion
from apply_filters import low_pass
from apply_filters import sort_pixels
from apply_filters import circlify
from apply_filters import circlify_movement


# Function that does nothing - used for trackbars
def nothing(x):
    pass


# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
cam = cv2.VideoCapture(0)
# cv2.CAP_DSHOW - add to remove black bars

# Track bars to control parameters
cv2.namedWindow("Controls")
cv2.createTrackbar("Threshold", "Controls", 0, 100, nothing)
cv2.createTrackbar("Sort axis", "Controls", 0, 2, nothing)
cv2.createTrackbar("Downsampling", "Controls", 10, 15, nothing)
# Initialize states for different filters
bw = False
th = False
motion = False
lp = False
sort = False
circle = False
circle_movement = False

# Tracks when filters are changed and when circlify is started
skip_frame = False
circlify_index = 0

# loops to capture, process and display frames
while True:
    # Get input image
    # ret - bool for successful/failed capture
    # raw_frame - image array
    ret, raw_frame = cam.read()

    # Make a copy of raw_frame for input to filters. Raw_frame will never be changed
    in_frame = raw_frame

    # Define output image for displaying - at this stage a copy of in_frame
    out_frame = in_frame

    # Get parameters for filters from track bars
    controller = cv2.getTrackbarPos("Threshold", "Controls") / 100
    sort_axis = cv2.getTrackbarPos("Sort axis", "Controls")
    sampling_factor = cv2.getTrackbarPos("Downsampling", "Controls")

    # Reset circlify counter when filter is turned off
    if not circle_movement:
        circlify_index = 0

    # Apply selected filters
    # bw filter is always applied first, so that other algorithms are run on it - modifies the in_frame
    if bw:
        out_frame = gray_scale(in_frame)
        in_frame = out_frame

    # Filters applied to the in_frame or, if they can be stacked, to the out_frame of previous filters
    if not skip_frame:

        if lp:
            out_frame = low_pass(controller, in_frame, p_frame_raw)

        if th:
            out_frame = threshhold(controller, in_frame)

        if motion:
            out_frame = detect_motion(in_frame, p_frame_raw)

        if sort:
            out_frame = sort_pixels(out_frame, sort_axis)

        if circle:
            out_frame = circlify(sampling_factor, out_frame)

        if circle_movement:
            # If this filter was just turned on, circlify whole image as a first step
            if circlify_index == 0:
                p_frame_processed = circlify(sampling_factor, out_frame)
            # Then use circlified image as background to apply  changes
            out_frame = circlify_movement(sampling_factor, out_frame, p_frame_raw, p_frame_processed)
            circlify_index += 1

    # display image

    final_image = np.concatenate((raw_frame, out_frame), axis = 1)
    print(final_image.shape)
    cv2.imshow("frame", final_image)
    # Save current frame for next loop - in raw and processed formats
    p_frame_raw = in_frame
    p_frame_processed = out_frame

    # Program control - key based
    # I really need to put most of these bools in a list to make this easier
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

    if key & 0xFF == ord('1'):
        bw = not bw
        print("Black and White" + str(bw))

    if key & 0xFF == ord('2'):
        th = not th
        print("Threshold" + str(th))

    if key & 0xFF == ord('3'):
        motion = not motion
        print("Motion detection" + str(motion))

    if key & 0xFF == ord('4'):
        lp = not lp
        print("Low pass filter" + str(lp))

    if key & 0xFF == ord('5'):
        sort = not sort
        print("Pixel sorting " + str(sort))

    if key & 0xFF == ord('6'):
        circle = not circle
        print("Circlification " + str(circle))

    if key & 0xFF == ord('7'):
        circle_movement = not circle_movement
        print("Circlification " + str(circle_movement))

    if key & 0xFF == ord('s'):
        bw = False
        th = False
        motion = False
        lp = False
        sort = False
        circle = False
        circle_movement = False
        print("All filters are now OFF")

    if key != -1:
        skip_frame = True

    if key == -1:
        skip_frame = False

cam.release()
cv2.destroyAllWindows()
