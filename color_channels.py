import cv2 as cv
import numpy as np

img = cv.imread('Photos/Guts.jpg')
cv.imshow('Guts', img)

#you can split a picture into its individual color channels
b,g,r = cv.split(img)

# These are now one channel pictures (gray scale), regions where there is a lot of blue in the original will look brighter in the blue (b) picture etc.
cv.imshow('Blue', b)
cv.imshow('Green', g)
cv.imshow('Red', r)

# Here you can see that the original img shape has 3 channels as the 3rd dimension and now the other pictures have no 3rd dimension.  
print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

# The merge function will create one multi-channel array out of several single-channel ones. So it will recreate the previous 
merged = cv.merge([b,g,r])
cv.imshow('Merged Image', merged)

# Instead of just seeing a gray scale image for each color channel we can reconstruct the image by creating a blank image
blank = np.zeros(img.shape[:2], dtype='uint8') # Creating a blank image of the width and heigth dimensions not the channels. Filling it with zeros

# We use the merge function, just like we did with the individual b,g,r single channel images, but instead we use the single channel images for the colors we don't want. 
blue = cv.merge([b, blank, blank])      
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('blue', blue)
cv.imshow('green', green)
cv.imshow('red', red)

cv.waitKey(0)