import cv2 as cv
import numpy as np

img = cv.imread("Photos/Guts.jpg")
cv.imshow("Original", img)

# _________________________________________________Contour DETECTION_________________________________
# 1. change to gray scale
# 2. find edges
# 3. Blur
# 4. use contours method

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Scale", gray)

canny = cv.Canny(gray, 125, 175)
cv.imshow("Canny Edges", canny)


contours, heirarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# The INPUTS (Parameters)
# canny: This is the source image. Crucially, this function must be given a binary image (single-channel, black and white). The variable name canny implies it's the output of an edge detection algorithm like cv.Canny, which is a perfect input. A simple thresholded image also works. You cannot pass a regular BGR color image directly.
# cv.RETR_LIST: This is the contour retrieval mode. It tells OpenCV how to organize the contours it finds, specifically their relationships to each other (e.g., if one shape is inside another). RETR_LIST is one of the simplest modes; it retrieves all contours but doesn't create any parent-child relationships. They are all returned as a flat list.
# cv.CHAIN_APPROX_NONE: This is the contour approximation method. It tells OpenCV how to store the points that make up the outline. CHAIN_APPROX_NONE stores every single point along the line of the contour. If you have a straight line that is 100 pixels long, this method will store all 100 points.

# THE OUTPUTS (Return Values)
# The function returns two values, which we are unpacking into the contours and hierarchies variables.
# (Note: In very old versions of OpenCV, it returned three values. This is a common point of confusion if you see old code online.)

# contours: This is a Python list containing all the individual contours that were found.

# Each individual contour in the list is a NumPy array of (x, y) coordinates that make up that specific outline.
# So, the structure is: [contour1, contour2, contour3, ...], where contour1 is [[x1,y1], [x2,y2], ...]
# hierarchies: This output is directly related to the retrieval mode (cv.RETR_...). It's a NumPy array that describes the parent-child relationships between contours. For each contour i, hierarchies[0][i] is an array of four values: [Next, Previous, First_Child, Parent].

# Next: Index of the next contour at the same hierarchical level.
# Previous: Index of the previous contour at the same hierarchical level.
# First_Child: Index of its first child contour.
# Parent: Index of its parent contour.
# A value of -1 means that element does not exist. With cv.RETR_LIST, this hierarchy information is not very useful as no parent-child links are established.

# Variations of the Retrieval Mode (cv.RETR_...)
# This choice depends on whether you care about shapes nested inside other shapes.

# cv.RETR_LIST (what you have): Gets all contours in a flat list. Good and simple if you don't care about relationships.
# cv.RETR_EXTERNAL: Most commonly used for simple cases. It only retrieves the "parent" or outermost contours and ignores any shapes that are inside another shape. If you just want to find the outline of each object, this is the one to use.
# cv.RETR_CCOMP: Retrieves all contours and organizes them into a two-level hierarchy. It groups all the external "parent" contours and the "hole" contours directly inside them.
# cv.RETR_TREE: Retrieves all contours and reconstructs the full, nested hierarchy of parent-child relationships. Use this if you need to analyze complex objects with multiple levels of nesting (e.g., a shape inside a hole that is inside another shape).

# Variations of the Approximation Method (cv.CHAIN_APPROX_...)
# This choice affects how much memory your contours use.
# cv.CHAIN_APPROX_NONE (what you have): Stores all points. Use this only if you need every single boundary point for a high-precision analysis, which is rare.
# cv.CHAIN_APPROX_SIMPLE: This is the most commonly used method. It dramatically compresses the contours by storing only the essential vertices. For a straight line, it will only store the two endpoints. For a rectangle, it will only store the four corner points. This saves a huge amount of memory and processing time with almost no loss of practical information.

print(f"{len(contours)} contour(s) found!")

blur = cv.GaussianBlur(
    canny, (5, 5), cv.BORDER_DEFAULT
)  # this will decrease how many contours we get.
cv.imshow("Blur", blur)
contours, heirarchies = cv.findContours(blur, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f"{len(contours)} contour(s) found!")

# another way to find contours instead of canny edge detector
ret, thresh = cv.threshold(
    gray, 125, 255, cv.THRESH_BINARY
)  # looks at an image and tries to binarize that image based on its intensity. Turns it into 0 or 1 white or black.
cv.imshow("thresh", thresh)
contours, heirarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f"{len(contours)} contour(s) found!")
# In short: Use Canny for finding edges. Use Thresholding for finding solid shapes and separating foreground from background.

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)

# gray: The input image, which must be grayscale.
# 125: This is the threshold value (thresh). This is your cutoff point.
# 255: This is the maximum value (maxval). It's the value that will be assigned to pixels that pass the threshold test.
# cv.THRESH_BINARY: This is the thresholding type. THRESH_BINARY means:
# If a pixel's intensity is greater than the threshold value (125), its value is set to the maxval (255, which is white).
# Otherwise, its value is set to 0 (black).
# The function returns two values:

# thresh: This is the resulting binary image (the black and white image).
# ret: This is the threshold value that was used. In this case, it will simply be 125. This return value is more useful for automatic thresholding methods (see variations below).
# 2. cv.imshow('thresh', thresh)

# This line simply displays the new black and white thresh image in a window so you can see the result of the operation.
# 3. contours, heirarchies = cv.findContours(...)

# This is the exact same function from our previous discussion. It takes the binary image thresh as input and finds the outlines of all the white areas.
# Notice the use of cv.CHAIN_APPROX_SIMPLE. As we discussed, this is a more efficient way to store the contours, as it only stores the vertices (e.g., the 4 corners of a square instead of all the points along its edges).
# Popular and Important Variations
# Hardcoding a value like 125 is often not ideal, as lighting conditions can change. OpenCV provides several powerful variations.

# 1. Different Thresholding Types
# Besides cv.THRESH_BINARY, you can use other types:

# cv.THRESH_BINARY_INV: The inverse operation. Pixels brighter than 125 become black (0), and darker pixels become white (255). This is perfect for finding dark objects on a light background.
# cv.THRESH_TRUNC: Truncate. Pixels brighter than 125 are set to 125. Pixels darker than 125 are unchanged. It's like putting a "ceiling" on the brightness.
# cv.THRESH_TOZERO: Pixels brighter than 125 are unchanged. Pixels darker than 125 are set to 0. This "zeros out" the dark parts of the image.
# 2. Adaptive Thresholding (Very Common!)
# Instead of one global threshold for the whole image, adaptive thresholding calculates a different threshold for different small regions of the image. This is extremely effective for images with varying lighting conditions (e.g., one side is in shadow).

# Python

# adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
#                                      cv.THRESH_BINARY, 11, 3)
# This is more complex, but it essentially calculates the threshold for a pixel based on the mean brightness of its small neighborhood of 11x11 pixels.

# 3. Otsu's Binarization (Extremely Powerful!)
# This is the most important variation to know. Instead of you guessing the best threshold value, Otsu's method automatically calculates the optimal threshold value by analyzing the histogram of the image. It assumes there are two peaks of brightness (background and foreground) and finds the best value to separate them.

# To use it, you combine the cv.THRESH_OTSU flag with another type and set your threshold value to 0.


# # Let OpenCV find the best threshold value for you!
# ret, otsu_thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# cv.imshow('Otsu Threshold', otsu_thresh)
# print(f"Otsu's method found the optimal threshold to be: {ret}")
# This is the go-to method when you need a robust, automatic way to binarize an image without manually tuning a threshold value.
# Draw the contours
blank_img = np.zeros(
    img.shape, dtype="uint8"
)  # here we make an np array of zeros and pass it the shape of img.Make each value a uint8
cv.drawContours(
    blank_img, contours, -1, (0, 0, 255), 1
)  # Pass the blank image the size of the original imgage, list of contours, how many contours we want (-1 is all of em), the color to draw, the thickness
cv.imshow("contours image", blank_img)


# draw the contours of canny
blank_img = np.zeros(
    img.shape, dtype="uint8"
)  # here we make an np array of zeros and pass it the shape of img.Make each value a uint8
contours_canny, heirarchies_canny = cv.findContours(
    canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
)
print(f"{len(contours_canny)} contour(s) found!")
cv.drawContours(
    blank_img, contours_canny, -1, (0, 0, 255), 1
)  # Pass the blank image the size of the original imgage, list of contours, how many contours we want (-1 is all of em), the color to draw, the thickness
cv.imshow("contours canny", blank_img)

cv.waitKey(0)
