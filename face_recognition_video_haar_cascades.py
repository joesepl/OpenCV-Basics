import cv2 as cv


# Create a Haar Cascade classifier for face detection
# Function: cv.CascadeClassifier(filename) -> CascadeClassifier
#   filename: path to the XML file containing the trained Haar cascade data (e.g., 'haar_face.xml')
# The classifier detects objects (like faces) in images using features learned from many positive and negative examples.
haar_cascade = cv.CascadeClassifier("haar_face.xml")

try:
    capture = cv.VideoCapture(0)
except:
    print("Error getting camera or video file.")
print("Video playing press q to quit.")
while True:
    isTrue, frame = capture.read()
    if isTrue:
        # Convert the image to grayscale for face detection (color is not needed)
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces in the image using the Haar Cascade classifier
        # Function: detectMultiScale(image, scaleFactor=1.1, minNeighbors=3, flags=0, minSize=None, maxSize=None) -> rectangles
        #   image: input image (usually grayscale)
        #   scaleFactor: how much the image size is reduced at each image scale (controls detection at different sizes)
        #   minNeighbors: how many neighbors each candidate rectangle should have to retain it (higher = fewer detections, more strict)
        #   flags: (usually 0, legacy parameter)
        #   minSize, maxSize: minimum and maximum possible object size (optional)
        # Returns: a list (numpy array) of rectangles, where each rectangle contains the coordinates (x, y, w, h) of a detected object (e.g., face)
        faces_rect = haar_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5
        )

        # Loop through all detected faces and draw rectangles around them
        cntr = 0
        for x, y, w, h in faces_rect:
            cv.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )  # Draw a rectangle around the face
            cntr += 1
        print(
            "Number of faces found: ", len(faces_rect)
        )  # Print the total number of faces detected

        # Show the image with rectangles drawn around detected faces
        cv.imshow("face", frame)

    if cv.waitKey(20) == ord("q"):
        break
capture.release()  # Release camera resource.
cv.destroyAllWindows()  # Close opencv windows.
