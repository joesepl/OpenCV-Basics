import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
img = cv.imread('Photos/Griffith.jpg')
cv.imshow('Guts', img)

# Create a blank mask (all zeros, same size as the image)
blank = np.zeros(img.shape[:2], dtype='uint8')
# Draw a filled white circle in the center of the mask
mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 200, 255, -1)
cv.imshow('mask', mask)

# Plotting the histogram using matplotlib
plt.figure()  # Create a new figure window
plt.title('Color Histogram')  # Set the plot title
plt.xlabel('Bins')  # Label for the x-axis (intensity values)
plt.ylabel('# of pixels')  # Label for the y-axis (number of pixels per bin)

# Color Histogram
# We'll plot the histogram for each color channel (Blue, Green, Red) separately.
colors = ('b', 'g', 'r')  # Tuple of color channel names (OpenCV uses BGR order)
for i, col in enumerate(colors):
    # Calculate the histogram for the current color channel
    # Function signature: cv.calcHist(images, channels, mask, histSize, ranges) -> hist
    #   images: list of images to process ([img])
    #   channels: index of the channel to calculate (0=Blue, 1=Green, 2=Red)
    #   mask: mask image to specify the region of interest (our circle)
    #   histSize: number of bins ([256] for 256 intensity levels)
    #   ranges: intensity range ([0,256])
    # Returns: the histogram as a numpy array
    hist = cv.calcHist([img], [i], mask, [256], [0,256])
    # Plot the histogram for this channel in its respective color
    plt.plot(hist, color = col)
    plt.xlim([0,256])  # Set the x-axis limits to match intensity range

plt.show()  # Display the plot window

cv.waitKey(0)