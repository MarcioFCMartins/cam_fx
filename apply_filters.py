# Filters to be applied to webcam input

# Some functions already present in OpenCV were re-implemented for learning purposes
# Includes some custom functions based on https://www.youtube.com/watch?v=mRM5Js3VLCk
# http://www.flong.com/texts/essays/essay_cvad/
# TODO Implement edge detection to learn convultions
# https://www.youtube.com/watch?v=uihBwtPIBxM
# https://www.youtube.com/watch?v=sRFM5IEqR2w
import cv2
import numpy as np
import math
import random


def gray_scale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def threshhold(value, frame):
    ret, frame = cv2.threshold(frame, int(round(value * 255)), 255, cv2.THRESH_BINARY)
    return frame


# The simplest motion detection possible is the absolute difference between frames
def detect_motion(current_frame, previous_frame):
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


def circlify(sampling_factor, current_frame):
    # Convert downsampling to 0.01-1 range
    sampling_factor = 1 if sampling_factor < 1 else sampling_factor
    sampling_factor = sampling_factor / 100

    # Downsample input frame
    ds_frame = cv2.resize(current_frame, None, fx=sampling_factor, fy=sampling_factor)

    # Create white array (8 bit integer) for background
    frame = np.full(current_frame.shape, 255, np.uint8)
    radius = round(current_frame.shape[1] / ds_frame.shape[1] / 2)
    # Loop over downsampled pixels, extract color. Drawn circles of correct color over white background
    for row in range(ds_frame.shape[1]):
        for col in range(ds_frame.shape[0]):
            if np.ndim(current_frame) == 3:
                color = tuple(ds_frame[col, row, :])
                color = (int(color[0]), int(color[1]), int(color[2]))
            elif np.ndim(current_frame) == 2:
                color = int(ds_frame[col, row])

            # Draw circle
            frame = cv2.circle(frame, (round(row / sampling_factor), round(col / sampling_factor)), radius, color, -1)

    return frame


def circlify_movement(sampling_factor, current_frame, previous_frame, previous_frame_processed):
    # Convert downsampling to 0.01-1 range
    sampling_factor = 1 if sampling_factor < 1 else sampling_factor
    sampling_factor = sampling_factor / 100

    # Downsample input frame
    ds_frame = cv2.resize(current_frame, None, fx=sampling_factor, fy=sampling_factor)

    # Create mask of pixels where movement occurred
    movement = detect_motion(current_frame, previous_frame)
    if np.ndim(movement) == 3:
        movement = cv2.cvtColor(movement, cv2.COLOR_BGR2GRAY)
    movement = cv2.resize(movement, None, fx=sampling_factor, fy=sampling_factor)
    ret, movement = cv2.threshold(movement, int(round(0.1 * 255)), 255, cv2.THRESH_BINARY)

    frame = previous_frame_processed

    # Calculate circle radius
    radius = round(current_frame.shape[1] / ds_frame.shape[1] / 2)

    # Loop over downsampled image, extract color. If movement occurred, draw circle corresponding to that pixel
    for row in range(ds_frame.shape[1]):
        for col in range(ds_frame.shape[0]):
            if movement[col, row] == 255:
                # Extract color
                if np.ndim(current_frame) == 3:
                    color = tuple(ds_frame[col, row, :])
                    color = (int(color[0]), int(color[1]), int(color[2]))
                elif np.ndim(current_frame) == 2:
                    color = int(ds_frame[col, row])

                # Create random noise for position
                noise_x = round((random.random() - 0.5) / (sampling_factor * 2))
                noise_y = round((random.random() - 0.5) / (sampling_factor * 2))
                # Draw circle
                frame = cv2.circle(frame,
                                   (round(row / sampling_factor) + noise_x, round(col / sampling_factor) + noise_y),
                                   radius, color, -1)

    return frame


def vectorify(sampling_factor, current_frame):
    # Create white array (8 bit integer) for background
    frame = np.full(current_frame.shape, 255, np.uint8)


    # Convert downsampling to 0.01-1 range
    sampling_factor = 1 if sampling_factor < 1 else sampling_factor
    sampling_factor = sampling_factor / 100
    # Downsample input frame
    ds_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    ds_frame = cv2.resize(ds_frame, None, fx=sampling_factor, fy=sampling_factor)

    radius = round(current_frame.shape[1] / ds_frame.shape[1] / 2)
    angle_factor = 255 / (math.pi/2)
    # Loop over downsampled pixels, extract color. Drawn circles of correct color over white background
    for row in range(ds_frame.shape[1]):
        for col in range(ds_frame.shape[0]):
            angle = ds_frame[col, row]/angle_factor
            center = (round(row / sampling_factor), round(col / sampling_factor))
            p1 = (int(radius * math.cos(angle) + center[0]), int(radius * math.sin(angle) + center[1]))
            p2 = (int(radius * math.cos(angle - math.pi) + center[0]), int(radius * math.sin(angle - math.pi) + center[1]))

            # Draw circle
            frame = cv2.line(frame, p1, p2, (0, 0, 0))

    return frame





