import os
import cv2 as cv
import numpy as np

# List of people (names) whose faces we want to recognize
people = ["Ben Afflek", "Elton John", "Jerry Seinfield", "Madonna", "Mindy Kaling"]
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
            # detectMultiScale
            # Parameters:
            #   gray: The input image (must be grayscale)
            #   scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
            #                (e.g., 1.1 means reduce size by 10% at each scale)
            #   minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
            #                 (Higher value results in fewer detections but with higher quality)
            faces_rect = haarcascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5
            )
            # For each face found, get the region of interest (ROI) and add it to features
            for x, y, w, h in faces_rect:
                faces_roi = gray[
                    y : y + h, x : x + w
                ]  # Crop the image to only contain the face.
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
# TO UNDERSTAND THIS ALGORITHM AND ITS PARAMETERS check the helpful algorith visualizers folder.
# Create the face recognizer (LBPH algorithm) object
face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)
# This function teaches the recognizer who is who. It requires two inputs that must be perfectly synchronized.
# features (the "What")
# What it is: A list of the images you want it to learn.
# Data Type: A standard Python list where every single item is a grayscale NumPy array representing a face.
# labels (the "Who")
# What it is: A list of ID numbers for each face in the features list.
# Data Type: A single NumPy array of integers (type np.int32).
# The Rule: The first image in the features list must correspond to the first number in the labels array, the second image to the second label, and so on.

face_recognizer.save("face_trained.yml")
# The save function takes the fully trained "brain"—all the calculated LBP histograms and their associated labels—and writes this complex data into a single file. This allows you to train your model once, save it, and then simply load that file in your final application without ever having to retrain.
# Can be saved as yaml or xml format

np.save("features.npy", features)
np.save("labels.npy", labels)
# np.save("features.npy", features) and np.save("labels.npy", labels): These lines save the INPUTS to the training. You are saving your pre-processed training data.

# Why? For convenience and future use. The process of detecting faces in all your images and organizing them into the features and labels lists can be slow. By saving these lists, you can:
# Retrain later: If you want to add more people, you don't need to re-scan all the old images. You can just load features.npy and labels.npy, add your new data, and then train a new model.
# Debug: You can load these files to inspect the exact data that went into your model if you're getting weird results.
