import cv2 as cv
import numpy as np

# Create a blank black image
blank = np.zeros((400,400), dtype='uint8')

# Draw a filled white rectangle on a copy of the blank image
# Function signature: cv.rectangle(img, pt1, pt2, color, thickness) -> img
#   img: image to draw on
#   pt1: top-left corner (x, y)
#   pt2: bottom-right corner (x, y)
#   color: color value (255 for white in grayscale)
#   thickness: thickness of the rectangle border (-1 for filled)
rectangle = cv.rectangle(blank.copy(), (30, 30), (370,370), 255, -1)

# Draw a filled white circle on a copy of the blank image
# Function signature: cv.circle(img, center, radius, color, thickness) -> img
#   img: image to draw on
#   center: center coordinates (x, y)
#   radius: radius of the circle
#   color: color value (255 for white in grayscale)
#   thickness: thickness of the circle border (-1 for filled)
circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

cv.imshow('rectangle', rectangle)
cv.imshow('circle', circle)


# BITWISE AND
# Function signature: cv.bitwise_and(src1, src2) -> dst
# Performs a pixel-wise AND operation between two arrays (images).
# For multi-channel (colored) images, the operation is applied independently to each channel (e.g., B, G, R).
# Each output pixel is 255 only if both corresponding input pixels are non-zero in that channel; otherwise, it is 0.
bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow('bitwise_and',bitwise_and)

# BITWISE OR
# Function signature: cv.bitwise_or(src1, src2) -> dst
# Performs a pixel-wise OR operation between two arrays (images).
# For colored images, the operation is applied independently to each channel.
# Each output pixel is 255 if at least one of the corresponding input pixels is non-zero in that channel; otherwise, it is 0.
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow('Bitwise Or', bitwise_or)

# BITWISE XOR
# Function signature: cv.bitwise_xor(src1, src2) -> dst
# Performs a pixel-wise XOR operation between two arrays (images).
# For colored images, the operation is applied independently to each channel.
# Each output pixel is 255 if only one of the corresponding input pixels is non-zero in that channel; otherwise, it is 0.
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow('xor', bitwise_xor)

# Inverts the color of one image
bitwise_not = cv.bitwise_not(rectangle)
cv.imshow('not', bitwise_not)


cv.waitKey(0)