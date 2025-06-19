import cv2 as cv
img = cv.imread('Guts')
cv.imshow('Original', img)

#_________________________________________________Contour DETECTION_________________________________

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Scale', gray)

canny = cv.Canny(gray, 125, 175)
cv.imshow('Canny Edges', canny)

