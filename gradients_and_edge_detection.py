import cv2 as cv
import numpy as np

# Edge Detection Algorithms Overview:
# - Laplacian: Good for detecting edges in all directions at once. Useful when you want a quick, general-purpose edge map, but it can be sensitive to noise.
# - Sobel: Calculates edges in a specific direction (x or y). Use Sobel when you want to analyze vertical or horizontal edges separately, or combine them for more control. Less sensitive to noise than Laplacian.
# - Canny: A multi-stage, advanced edge detector. It includes noise reduction, edge thinning, and edge linking. Use Canny when you need precise, clean edge maps and can tune parameters for best results. It is widely used in real-world applications.
# Choose the algorithm based on your needs: Laplacian for simplicity, Sobel for directional analysis, and Canny for accuracy and robustness.

# Gradients in image processing represent the change in intensity or color in an image.
# They are used to detect edges, as edges are locations with a sharp change in intensity.
# Edge detection is important for object detection, segmentation, and computer vision tasks.

# Read the input image
img = cv.imread("Photos/Griffith.jpg")
cv.imshow("Griffith", img)

# Convert the image to grayscale for edge detection (works on single-channel images)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)

# Laplacian Edge Detection
# The Laplacian operator calculates the second derivative of the image, highlighting regions where the intensity changes rapidly (edges).
# It combines information from both the x and y directions, making it sensitive to all edge orientations.
# Function signature: cv.Laplacian(src, ddepth[, ksize[, scale[, delta[, borderType]]]]) -> dst
#   src: input image (grayscale)
#   ddepth: desired depth of the output image (e.g., cv.CV_64F for high precision)
#   ksize: aperture size used to compute the second-derivative filters (default is 1)
#   scale, delta, borderType: optional parameters for scaling, offset, and border handling
# Returns: the Laplacian edge-detected image
# The Mechanic: The Laplacian operator works by calculating the second derivative of the image intensity at each pixel.
# Instead of just looking for a change (like the first derivative), it looks for places where the rate of change itself changes sharply—these are often edges.
# It does this by convolving the image with a small kernel that sums the differences in both the x and y directions, making it sensitive to all edge orientations.
lap = cv.Laplacian(gray, cv.CV_64F)
# Take the absolute value and convert to 8-bit for display
# The Laplacian result can have negative values (since edges can be positive or negative changes).
# np.absolute(lap) makes all values positive so all edges are visible.
# np.uint8(...) converts the result to 8-bit unsigned integers (0-255), which is the standard format for displaying images in OpenCV.
lap = np.uint8(np.absolute(lap))
cv.imshow("Laplacian", lap)

# Sobel Edge Detection
# The Sobel operator calculates the first derivative of the image intensity, highlighting regions where the intensity changes (edges).
# It can be used to detect edges in the x direction (vertical edges) or y direction (horizontal edges).
# Function signature: cv.Sobel(src, ddepth, dx, dy[, ksize[, scale[, delta[, borderType]]]]) -> dst
#   src: input image (grayscale)
#   ddepth: desired depth of the output image (e.g., cv.CV_64F)
#   dx: order of the derivative in the x direction (1 for Sobel x, 0 for Sobel y)
#   dy: order of the derivative in the y direction (0 for Sobel x, 1 for Sobel y)
#   ksize, scale, delta, borderType: optional parameters for kernel size, scaling, offset, and border handling
# Returns: the Sobel edge-detected image
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)  # Detects vertical edges
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)  # Detects horizontal edges

cv.imshow("sobelx", sobelx)
cv.imshow("sobely", sobely)

# Combine the two Sobel results to get edges in both directions
combined_sobel = cv.bitwise_or(sobelx, sobely)
cv.imshow("combined sobel", combined_sobel)

# Canny Edge Detection
# The Canny algorithm is a multi-stage edge detection process that is very effective and widely used.
# It involves noise reduction, gradient calculation, non-maximum suppression, and edge tracking by hysteresis.
# Function signature: cv.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) -> edges
#   image: input image (grayscale)
#   threshold1: lower threshold for edge detection
#   threshold2: upper threshold for edge detection
#     In Canny, the lower and upper thresholds are used for edge linking by hysteresis:
#     - Pixels with a gradient above the upper threshold are considered strong edges.
#     - Pixels below the lower threshold are suppressed (not edges).
#     - Pixels between the two are considered edges only if they are connected to strong edge pixels.
#     This helps keep only real, connected edges and reduces noise or weak, isolated responses.
#   apertureSize, L2gradient: optional parameters for kernel size and gradient calculation
# Returns: a binary image with detected edges
canny = cv.Canny(gray, 150, 175)
cv.imshow("Canny", canny)

cv.waitKey(0)
