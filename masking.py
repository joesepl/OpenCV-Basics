import cv2 as cv
import numpy as np

# Read the input image
img = cv.imread('Photos/Guts.jpg')
cv.imshow('Guts', img)

# Create a blank (black) mask with the same height and width as the image
# The mask must match the image's first two dimensions (height, width)
blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('Blank Image', blank)

# Draw a filled white circle in the center of the mask
# Only the region inside this circle will be affected by the mask
mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), img.shape[1]//2, 255, -1)
cv.imshow('mask', mask)

# Bitwise AND with mask: shows the original image only where the mask is white (255), black elsewhere
bitwise_and = cv.bitwise_and(img, img, mask=mask)


cv.imshow('bitwise and', bitwise_and)


cv.waitKey(0)