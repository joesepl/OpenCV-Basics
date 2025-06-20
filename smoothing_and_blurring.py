import cv2 as cv

# Read the image from the specified path
img = cv.imread('Photos/Griffith.jpg')
cv.imshow('Guts', img)

# Averaging Blur:
# Function signature: cv.blur(src, ksize) -> dst
#   src: input image
#   ksize: tuple specifying the width and height of the kernel (e.g., (7,7))
#   dst: output (blurred) image
# This method uses a simple mean filter. It defines a kernel window (in this case, 7x7),
# slides it over the image, and replaces the center pixel with the average intensity of all pixels inside the window.
# The higher the kernel size, the more pronounced the blur effect.
average = cv.blur(img, (7,7))
cv.imshow('Average Blur', average)

# Gaussian Blur:
# Function signature: cv.GaussianBlur(src, ksize, sigmaX) -> dst
#   src: input image
#   ksize: tuple specifying the width and height of the kernel (must be odd and positive)
#   sigmaX: standard deviation in X direction (0 means calculated from kernel size)
#   dst: output (blurred) image
# Unlike averaging, Gaussian blur uses a kernel with values following a Gaussian distribution (bell curve).
# This means pixels closer to the center of the kernel are given more weight than those farther away.
# As a result, Gaussian blur preserves edges better and produces a more natural, less harsh blur compared to simple averaging.
gaussian = cv.GaussianBlur(img, (7,7), 0)
cv.imshow('Gaussian Blur', gaussian)


# Median Blur:
# Function signature: cv.medianBlur(src, ksize) -> dst
#   src: input image
#   ksize: size of the kernel (must be odd and greater than 1)
#   dst: output (blurred) image
# This method replaces each pixel's value with the median of the intensities in the kernel window.
# It is especially effective at removing 'salt and pepper' noise while preserving edges better than averaging.
# Not typically used with large kernel sizes.
median = cv.medianBlur(img, 7)
cv.imshow('Median Blur', median)

# Bilateral Blur:
# Function signature: cv.bilateralFilter(src, d, sigmaColor, sigmaSpace) -> dst
#   src: input image
#   d: diameter of each pixel neighborhood
#   sigmaColor: filter sigma in the color space (how much colors within the window influence each other)
#   sigmaSpace: filter sigma in the coordinate space (how far pixels influence each other)
#   dst: output (blurred) image
# Bilateral filtering is unique because it considers both spatial proximity and pixel intensity similarity when blurring.
# This means that only pixels that are both close in space and similar in color to the center pixel will significantly influence the blur.
# As a result, bilateral filtering smooths flat regions while preserving sharp edges, making it ideal for tasks like noise reduction without losing edge detail or for stylized effects like cartoonization.
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('bilateral', bilateral)

cv.waitKey(0)



