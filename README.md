# Webcam effects

This is a Python learning project. The goals of this app were to:

1. Learn python in general
2. Learn numpy array operations
3. Learn OpenCV basics for computer vision

### Instructions

When run, 2 windows appear. One with the processed camera image and one with sliders to control applied filters

**Current filters** 

The filters are turned on/off by pressing the number keys. 

1. Grayscale - OpenCV function
2. Threshold - OpenCV function, controlled by the slider "Threshold"
3. Motion detector - Custom implementation, controlled by the slider "Threshold"
4. Low pass filter - Custom implementation, controlled by the slider "Threshold"
5. Sort pixels by color - Custom implementation, sorts by values in HSV color space. The dimension used for sorting is controlled by the slider "Sort axis"
6. Turn image into a circle grid. Circle size can be controlled by the slider "Downsampling"
7. Similar to previous, but circles have some randomness in their position and circles are only drawn for areas where movement has been detected. Doesn't look great because it relies on the motion detection I implemented but still interesting.

q. Exit application
s. Stop all filters