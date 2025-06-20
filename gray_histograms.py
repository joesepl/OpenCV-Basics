import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
img = cv.imread('Photos/Griffith.jpg')
cv.imshow('Guts', img)

# Convert the image to grayscale for histogram analysis
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Create a blank mask (all zeros, same size as the image)
blank = np.zeros(img.shape[:2], dtype='uint8')
# Draw a filled white circle in the center of the mask
circle = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 200, 255, -1)
# Apply the circular mask to the grayscale image using bitwise AND
# Only the region inside the circle will be visible in the mask result
gray_mask = cv.bitwise_and(gray, gray, mask=circle)
cv.imshow('Gray_Mask', gray_mask)

# Histogram: A histogram is a graphical representation of the distribution of pixel intensities in an image.
# It shows how many pixels have each possible intensity value (0-255 for grayscale).
# Histograms are used for image analysis, contrast adjustment, thresholding, and more.
# If you use the guts image you will see most of the values are close to 0 because its a darker image. Lots of blacks. Whereas the griffith image is more balanced. 

# Calculate the grayscale histogram
# Function signature: cv.calcHist(images, channels, mask, histSize, ranges) -> hist
#   images: list of images to process (e.g., [gray])
#   channels: list of channel indices to calculate the histogram for ([0] for grayscale)
#   mask: optional mask image (None means use the whole image)
#   histSize: number of bins (e.g., [256] for 256 intensity levels)
#     A bin is a range of intensity values grouped together in the histogram. For an 8-bit grayscale image, each bin typically represents one possible pixel value (0-255), so 256 bins means each bin counts the number of pixels with a specific intensity.
#   ranges: intensity range (e.g., [0,256])
# Returns: the histogram as a numpy array

gray_hist = cv.calcHist([gray], [0], gray_mask, [256], [0,256])


# Plotting the histogram using matplotlib
plt.figure()  # Create a new figure window
plt.title('Grayscale Histogram')  # Set the plot title
plt.xlabel('Bins')  # Label for the x-axis (intensity values)
plt.ylabel('# of pixels')  # Label for the y-axis (number of pixels per bin)
plt.plot(gray_hist)  # Plot the histogram data as a line graph
plt.xlim([0,256])  # Set the x-axis limits to match intensity range
plt.show()

cv.waitKey(0)