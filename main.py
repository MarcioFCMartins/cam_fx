import cv2
from apply_filters import gray_scale
from apply_filters import threshhold
from apply_filters import detect_motion
from apply_filters import low_pass
from apply_filters import sort_pixels


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
# Initialize states for different filters
bw = False
th = False
motion = False
lp = False
# Quantization, currently only sorts the pixels based on largest range axis
sort = False

# loops to capture, process and display frames until q is pressed
# in_frame = current capture from cam
# p_frame = last capture from the cam
# out_frame = image to be displayed
skip_frame = False

while True:
    # Get input image
    # ret - bool for successful/failed capture
    # frame - image array
    ret, in_frame = cam.read()

    # Define output image (currently no processing)
    out_frame = in_frame

    # Get parameters for filters from track bars
    controller = cv2.getTrackbarPos("Threshold", "Controls") / 100
    sort_axis = cv2.getTrackbarPos("Sort axis", "Controls")

    # Apply selected filters
    # bw filter is always applied first, so that other algorithms are run on it
    if bw:
        out_frame = gray_scale(in_frame)
        in_frame = out_frame

    # Some filters will crash when transitioning bw - color. Skip them for one frame
    if not skip_frame:

        if lp:
            out_frame = low_pass(controller, in_frame, p_frame)

        if th:
            out_frame = threshhold(controller, in_frame)

        if motion:
            out_frame = detect_motion(controller, in_frame, p_frame)

        if sort:
            out_frame = sort_pixels(out_frame, sort_axis)

    # display image
    cv2.imshow("frame", out_frame)
    # Save current frame for next loop
    p_frame = in_frame

    # Program control - key based
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

    if key & 0xFF == ord('1'):
        bw = not bw
        print("Black and White" + str(bw))

    if key & 0xFF == ord('2'):
        th = not th
        print("Tresholding" + str(th))

    if key & 0xFF == ord('3'):
        motion = not motion
        print("Motion detection" + str(motion))

    if key & 0xFF == ord('4'):
        lp = not lp
        print("Low pass filter" + str(lp))

    if key & 0xFF == ord('5'):
        sort = not sort
        print("Pixel sorting " + str(sort))

    if key != -1:
        skip_frame = True

    if key == -1:
        skip_frame = False



cam.release()
cv2.destroyAllWindows()
