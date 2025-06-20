import cv2 as cv
import numpy as np

# _________________________TRANSFORMATIONS________________________________

img = cv.imread("Photos/Guts.jpg")

cv.imshow("Guts", img)


# TRANSLATION
def translate(img, x, y):
    # x how many pixes to translate to the x axis and y is how many to translate on the y
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    print(transMat)
    dimensions = (
        img.shape[1],
        img.shape[0],
    )  # this gets the height,width (page,row) of the picture. It reverses the normal order of hieght, width to width, height. It doesn't do anything with the columms which are the channels.
    return cv.warpAffine(img, transMat, dimensions)

    # transMat = np.float32([[1,0,x], [0,1,y]]) #positive y values shift it down, -y up, +x right, -x left
    # This is the most important line in the function. It creates the transformation matrix that defines how the shift should happen.
    # What is a transformation matrix? In computer vision, a 2x3 matrix can describe several types of 2D "affine" transformations, including translation, rotation, and scaling, all at once.
    # Breaking down this specific matrix:
    # The matrix has the form: [[1, 0, x], [0, 1, y]]
    # The [[1, 0], [0, 1]] part is the identity matrix for rotation and scaling. It essentially means: "Don't rotate the image, and don't scale it (keep it at 100% size)." Just like in math if you change the identity vectors they will transform the graph. We kept them normal here. 1,0 for ihat or x and 0,1 for jhat y
    # The [x, y] part is the translation vector. The x value dictates the horizontal shift, and the y value dictates the vertical shift.
    # np.float32(...): This converts the matrix into a NumPy array with the data type float32. OpenCV's transformation functions expect their matrices to be in this specific floating-point format. Each value representing 0-255 only positive numbers.

    # warpAffine applies an affine transformation to an image. It takes the original image array, the 2x3 transformation matrix that tells the pixels how to move, Dimensions defines the size of the canvas that the shifted image will be drawn onto.


translated = translate(img, -100, 100)  # translate left 100 and up 100
cv.imshow("Translated", translated)


# ROTATION
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[
        :2
    ]  # shape returns a dimension tuple like (100,100,3). For us we only care about the height and width of the image so we only get 100, and 100 of the tuple

    if rotPoint is None:
        rotPoint = (
            width // 2,
            height // 2,
        )  # set rotation point to the midway point of width and height

    # This is the core of the rotation logic. It creates the specific 2x3 transformation matrix needed for this operation.
    # cv.getRotationMatrix2D(...) is an OpenCV helper function that builds this matrix for you. It's much easier than creating it manually with sines and cosines.
    # It takes three parameters:
    # rotPoint: The (x, y) coordinate to rotate the image around.
    # angle: The angle of rotation in DEGREES (not radians). This is a very important detail, as the comment in the original code was incorrect.
    # 1.0: This is a scale factor. A value of 1.0 means the rotated image will have the same size as the original (100% scale). If you passed 0.5, the image would be rotated and shrunk by 50%.

    rotMat = cv.getRotationMatrix2D(
        rotPoint, angle, 1.0
    )  # parameters: roation point, angle in radians, and scale.
    dimensions = (width, height)
    return cv.warpAffine(img, rotMat, dimensions)


rotated = rotate(img, 45)
cv.imshow("rotated", rotated)


# RESIZE
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow("Resized", resized)

# Flipping
flip = cv.flip(
    img, -1
)  # flip code can be 0, 1, or -1. Flip image verticallly (0), flip it horizontally (1), flip it horizontally and vertically (-1)
cv.imshow("flip", flip)

# Cropping
cropped = img[200:400, 300:400]
cv.imshow("cropped", cropped)

cv.waitKey(0)
