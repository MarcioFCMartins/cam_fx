# Filters to be applied to webcam input

# Some functions already present in OpenCV were re-implemented for learning purposes
# Includes some custom functions based on https://www.youtube.com/watch?v=mRM5Js3VLCk
# Color quantization based on my old R project https://github.com/MarcioFCMartins/Color_Quantization
# http://www.flong.com/texts/essays/essay_cvad/
import cv2
import numpy as np


def gray_scale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def threshhold(value, frame):
    ret, frame = cv2.threshold(frame, int(round(value * 255)), 255, cv2.THRESH_BINARY)
    return frame


# The simplest motion detection possible is the absolute difference between frames
def detect_motion(value, current_frame, previous_frame):
    frame = cv2.absdiff(current_frame, previous_frame)
    return frame


def low_pass(value, current_frame, previous_frame):
    # If color
    if np.ndim(current_frame) == 3:
        layers = range(current_frame.shape[2])
        frame_shape = current_frame.shape
        current_array = np.empty((frame_shape[0] * frame_shape[1], frame_shape[2]))
        for i in layers:
            current_array[:, i] = current_frame[:, :, i].flatten()

        previous_array = np.empty((frame_shape[0] * frame_shape[1], frame_shape[2]))
        for i in layers:
            previous_array[:, i] = previous_frame[:, :, i].flatten()
    # If grayscale
    elif np.ndim(current_frame) == 2:
        current_array = current_frame.flatten()
        previous_array = previous_frame.flatten()

    change = current_array - previous_array
    change = change * value
    current_array = previous_array + change

    # If color
    if np.ndim(current_frame) == 3:
        layers = range(current_frame.shape[2])
        for i in layers:
            current_frame[:, :, i] = current_array[:, i].reshape((480, 640))
    # If grayscale
    elif np.ndim(current_frame) == 2:
        current_frame = current_array.reshape((480, 640))

    return current_frame


def sort_pixels(current_frame, axis):
    # If color image
    if np.ndim(current_frame) == 3:
        # Convert to HSV for sorting
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # Convert to 2-dimensional array
        frame_array = np.column_stack([
            current_frame[:, :, 0].flatten(),
            current_frame[:, :, 1].flatten(),
            current_frame[:, :, 2].flatten()]
        )

        # Sort by selected axis
        sorted_frame = frame_array[np.argsort(frame_array[:, axis])]

        # Shape back to 3 dimensions
        current_frame[:, :, 0] = sorted_frame[:, 0].reshape((480, 640))
        current_frame[:, :, 1] = sorted_frame[:, 1].reshape((480, 640))
        current_frame[:, :, 2] = sorted_frame[:, 2].reshape((480, 640))

        # Convert to BGR for plotting
        current_frame = cv2.cvtColor(current_frame, code=cv2.COLOR_HSV2BGR)
    # If grayscale image
    elif np.ndim(current_frame) == 2:
        frame_array = current_frame.flatten()

        sorted_frame = frame_array[frame_array.argsort()]

        current_frame = sorted_frame.reshape((480, 640))

    return current_frame
