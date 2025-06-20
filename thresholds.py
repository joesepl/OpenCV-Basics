import cv2 as cv

# Thresholding is a technique in image processing used to separate objects from the background.
# It works by converting a grayscale image into a binary image: pixels above a certain value (the threshold) are set to one value (e.g., white), and all others are set to another (e.g., black).
# In basic (global) thresholding, a single threshold value is chosen and applied to all pixels in the image.
# In adaptive thresholding, the threshold value is calculated for smaller regions of the image, allowing for better results with varying lighting conditions.
# Thresholding is useful for object detection, segmentation, and preparing images for further analysis.

# Read the input image
img = cv.imread("Photos/Griffith.jpg")
cv.imshow("Griffith", img)

# Convert the image to grayscale (thresholding works on single-channel images)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

# Simple Thresholding
# Function signature: cv.threshold(src, thresh, maxval, type) -> retval, dst
#   src: input grayscale image
#   thresh: threshold value
#   maxval: value to set for pixels above the threshold
#   type: thresholding type (e.g., cv.THRESH_BINARY for standard binary thresholding, cv.THRESH_BINARY_INV for inverted, cv.THRESH_TRUNC, cv.THRESH_TOZERO, cv.THRESH_TOZERO_INV for other thresholding behaviors)
# Returns: the threshold value used and the thresholded image
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow("Simple Thresholded", thresh)

threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow("Simple Thresholded Inverse", thresh_inv)

# Adaptive Thresholding
# Unlike simple thresholding, adaptive thresholding calculates the threshold for smaller regions of the image.
# This is useful when lighting conditions are not uniform across the image.
# The Mechanic: Instead of using one global value for the whole image, adaptive thresholding calculates a different, unique threshold for each small region of the image.
# For every single pixel in the image, OpenCV looks at a small neighborhood around it (e.g., an 11x11 square).
# It calculates a threshold value based only on the pixels within that small neighborhood.
# It then applies this unique, local threshold to the center pixel.
# Function signature: cv.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C) -> dst
#   src: input grayscale image
#   maxValue: value to assign if condition is met (e.g., 255)
#   adaptiveMethod: algorithm to use (cv.ADAPTIVE_THRESH_MEAN_C or cv.ADAPTIVE_THRESH_GAUSSIAN_C)
#   thresholdType: type of thresholding (e.g., cv.THRESH_BINARY, cv.THRESH_BINARY_INV)
#   blockSize: size of the neighborhood area used to calculate the threshold for each pixel (must be odd)
#   C: constant subtracted from the mean or weighted mean
# Returns: the thresholded image
adaptive_thresh = cv.adaptiveThreshold(
    gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 5
)
cv.imshow("Adaptive Thresholding", adaptive_thresh)

adaptive_thresh_Gaussian = cv.adaptiveThreshold(
    gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 5
)
cv.imshow("Adaptive Thresholding Gaussian", adaptive_thresh_Gaussian)

cv.waitKey(0)
