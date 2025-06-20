import cv2 as cv

img = cv.imread("Photos/Guts.jpg")
cv.imshow("Guts", img)

# # Converting to grayscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Guts_BW', gray)

# # Blur
# blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)   #Blurs an image by removing noise. To increase blur increase kernel size from 5,5
# cv.imshow('Blur', blur)

# # Create edge Cascade (cany edge detectory)
# canny = cv.Canny(img, 125, 175)
# cv.imshow('edges', canny )

# # Create edge cascade from blurred image to get less edges
# canny_blur = cv.Canny(blur, 125, 175)
# cv.imshow('blurred edges', canny_blur)

# # Dilate an image using an element
# dilated = cv.dilate(canny, (7,7), iterations=3)     # to increase dilation (7,7), iterations will also increase dilation by ieterating.
# cv.imshow('dilated', dilated)

# # Eroding, takes a dilating image and tries to return a nondilated image closer to the normal image
# eroded = cv.erode(dilated, (7,7), iterations=3 )
# cv.imshow('eroded', eroded)

# Resize Image
resize = cv.resize(img, (500, 1280))  # resizes image ignoring aspect ratio
cv.imshow("resized", resize)

# Resize Image Keeping Aspect Ratio
resized_with_ratio = cv.resize(
    img, (300, 150), interpolation=cv.INTER_AREA
)  # use interpolation cv.INTER_AREA for shrinking an image. cv.INTER_CUBIC for highest quality but slowest processing scaling image to be bigger or cv.LINEAR for a faster lower quality enlarging interpolation
cv.imshow("resized ratio", resized_with_ratio)

# Resize Image Keeping Aspect Ratio - Enlarging
resized_ratio_enlarge = cv.resize(
    img, (500, 1280), interpolation=cv.INTER_CUBIC
)  # supposedly better for scaling
cv.imshow("intercubic enlarge", resized_ratio_enlarge)

# Cropping
cropped = img[50:200, 200:400]
cv.imshow("cropped", cropped)

cv.waitKey(0)
