import cv2 as cv

# Read the input image
img = cv.imread("Photos/family.jpg")
cv.imshow("face", img)

# Convert the image to grayscale for face detection (color is not needed)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("face", gray)

# Create a Haar Cascade classifier for face detection
# Function: cv.CascadeClassifier(filename) -> CascadeClassifier
#   filename: path to the XML file containing the trained Haar cascade data (e.g., 'haar_face.xml')
# The classifier detects objects (like faces) in images using features learned from many positive and negative examples.
haar_cascade = cv.CascadeClassifier("haar_face.xml")

# Detect faces in the image using the Haar Cascade classifier
# Function: detectMultiScale(image, scaleFactor=1.1, minNeighbors=3, flags=0, minSize=None, maxSize=None) -> rectangles
#   image: input image (usually grayscale)
#   scaleFactor: how much the image size is reduced at each image scale (controls detection at different sizes)
#   minNeighbors: how many neighbors each candidate rectangle should have to retain it (higher = fewer detections, more strict)
#   flags: (usually 0, legacy parameter)
#   minSize, maxSize: minimum and maximum possible object size (optional)
# Returns: a list (numpy array) of rectangles, where each rectangle contains the coordinates (x, y, w, h) of a detected object (e.g., face)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Loop through all detected faces and draw rectangles around them
cntr = 0
for num_of_faces in faces_rect:
    print(
        "num of faces", num_of_faces
    )  # Print the coordinates and size of each detected face
    x, y, w, h = faces_rect[cntr, 0:4]  # Get the position and size of the current face
    cv.rectangle(
        img, (x, y), (x + w, y + h), 255, 2
    )  # Draw a rectangle around the face
    cntr += 1
print(
    "Number of faces found: ", len(faces_rect)
)  # Print the total number of faces detected

# Show the image with rectangles drawn around detected faces
cv.imshow("face", img)

cv.waitKey()
