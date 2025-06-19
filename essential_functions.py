import cv2 as cv

img = cv.imread('Photos/Guts.jpg')
cv.imshow('Guts', img)

#converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Guts_BW', gray)

cv.waitKey(0)