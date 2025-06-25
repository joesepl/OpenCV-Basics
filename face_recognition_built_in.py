import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier(
    "haar_face.xml"
)  # Load the haar_cascade for face recognition

# Load the features, labels, and names list in case we want to append to it more face data.
people = ["Ben Afflek", "Elton John", "Jerry Seinfield", "Madonna", "Mindy Kaling"]
features = np.load(
    "features.npy", allow_pickle=True
)  # A list of the cropped portion of the images where a face has been detected.
labels = np.load("labels.npy", allow_pickle=True)


# Create a LBH face recognizer object and Load the trained face recognition data (the brain)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

# Read an image of a person to validate it and convert to gray
img = cv.imread('Faces_validation\madonna/5.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# First detect the face(s), then crop the images based on the square sent back from the face detection. 
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
for(x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    # Predict the current face by passing the cropped out face to the predict function of our face recongizer.
    label, confidence = face_recognizer.predict(faces_roi)

    # Print the persons name by checking the label (some index) and returning the name for that index. Also print the confidence.
    print(f'Label = {people[label]} with a confidence of {confidence}')

    #print a label of the person that the image is showing. Also print a rectangle over the face. 
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)
cv.waitKey(0)