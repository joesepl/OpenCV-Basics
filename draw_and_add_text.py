import cv2 as cv
import numpy as np

# blank = np.zeros((500,500), dtype='uint8') #create a blank 500 by 500 image of data type uint8 (holds values from 0-255 unsigned (no negatives)). This is a single channel image. A gray scale image.
blank = np.zeros(
    (500, 500, 3), dtype="uint8"
)  # create a blank 500 by 500 image of data type uint8. It has 3 channels (columns) for bgr except its ordered bgr.
print(blank)

# 1. Paint the image a certain color
# blank[:] = [0,255,0]        #for every page and every row, the 3 dimensional column will be filled with 0,255,0. Paint green.
# blank[:] = [0,0,255]        #paint it red
# blank[20:40,20:40]  = [255,0,0] #paint blue square manually

# 2. Draw rectangle with rectangle fmethod
cv.rectangle(
    blank, (0, 0), (250, 250), (0, 255, 0), thickness=cv.FILLED
)  # cv.rectangle(image_array, start_coord, end_coord, bgr_color, thickness=2). Filled keyword can also be -1. This will fill the whole rectangle.

# 3. Draw circle with circle method
center_x = int(blank.shape[0] / 2)  # find x center point by dividing width by 2.
center_y = int(blank.shape[1] / 2)  # find y center point by dividing height by 2.
radius = int(
    blank.shape[0] / 2
)  # find radius to fill full page by dividing width by 2.
cv.circle(blank, (center_x, center_y), (radius), (255, 0, 0), thickness=2)

# Another way to write the circle
# cv.circle(blank, (blank.shape[0]//2, blank.shape[1]//2), (blank.shape[0]//2), (255,0,0), thickness=2)  # the // is necessary because when u divide because dividing in python automatically returns a float unless you use //.


# 4. Draw Line
cv.line(
    blank, (blank.shape[0], blank.shape[1]), (0, 0), (255, 255, 0), thickness=2
)  # line(image_array, (start_coord_x, start_coord_y), (end_coord_x, end_coord_y), (b,g,r), thickness=2)


# 5. Write text
cv.putText(
    blank, "Hello", (0, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 0, 0), thickness=3
)  # putText(img, text, coord, fontFace, fontscale, color, thickenss=None)

cv.imshow("Canvas", blank)

cv.waitKey(0)
