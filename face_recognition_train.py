import os
import cv2 as cv
import numpy as np

# List of people (names) whose faces we want to recognize
people = ["Ben Afflek", "Elton John", "Jerry Seinfield", "Madonna", "Mindy Kaling"]
p = []  # This variable is not used, can be ignored
# Path to the folder where the training face images are stored
DIR = r"C:\Users\Jojo\Documents\Python Programming Track -Gemini\OpenCV Projects\OpenCV-Basics\Faces_train"
# Load the Haar Cascade classifier for face detection
haarcascade = cv.CascadeClassifier("haar_face.xml")

# Lists to store the face data (features) and their corresponding labels (person index)
features = []
labels = []


# Function to create the training data
# It will go through each person's folder, read each image, detect the face, and store the face region and label
def create_train():
    # Loop through each person in the people list
    for person in people:
        # Get the path to this person's folder
        path = os.path.join(DIR, person)
        # The label is just the index of the person in the people list
        label = people.index(person)

        # Loop through each image file in this person's folder
        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            # Read the image from disk
            img_array = cv.imread(img_path)
            # Convert the image to grayscale (face detection works better on grayscale)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces_rect = haarcascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4
            )
            # For each face found, get the region of interest (ROI) and add it to features
            for x, y, w, h in faces_rect:
                faces_roi = gray[y : y + h, x : x + w]
                features.append(faces_roi)  # Store the face region
                labels.append(label)  # Store the label (person index)


# Call the function to create the training data
create_train()
# Print how many faces (features) and labels were found
print(f"Length of the features = {len(features)}")
print(f"Length of the labels = {len(labels)}")
# Convert the features list to a numpy array (required for training)
features = np.array(features, dtype="object")
print("Training Done")
# Convert the labels list to a numpy array
labels = np.array(labels)

# Create the face recognizer (LBPH algorithm)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
# Train the recognizer on the features and labels
face_recognizer.train(features, labels)

face_recognizer.save("face_trained.yml")
# Save the features and labels arrays to disk for later use
np.save("features.npy", features)
np.save("labels.npy", labels)
