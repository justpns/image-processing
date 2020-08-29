# import the necessary packages
import cv2

# used to access local directory
import os

# used to plot our images
import matplotlib.pyplot as plt
import numpy as np

faceDet = cv2.CascadeClassifier(
    "./haar_models/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier(
    "./haar_models/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier(
    "./haar_models/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier(
    "./haar_models/haarcascade_frontalface_alt_tree.xml")

cv2.namedWindow("preview")
capture = cv2.VideoCapture(0)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    ret, frame = capture.read()
    image = frame.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceDet.detectMultiScale(
        gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

    face_two = faceDet_two.detectMultiScale(
        gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

    face_three = faceDet_three.detectMultiScale(
        gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

    face_four = faceDet_four.detectMultiScale(
        gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
    # Go over detected faces, stop at first detected face, return empty if no face.

    if len(faces) == 1:
        facefeatures = faces
    if len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        facefeatures = ""
        print("No Faces Detected")

    # Print coordinates of detected faces
    # print("Faces:\n", facefeatures)

	# # loop over the face detections
    for (x, y, w, h) in facefeatures:
        cropped_gray_face = gray_image[y:y + h, x:x + w]
        cropped_original_face = image[y:y + h, x:x + w]

		# show the face number
        cv2.putText(image, "Face #{}".format(x), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.rectangle(image,(x, y),(x + w,y + h),(0, 255, 0), 2)

    cv2.imshow('Detected Faces', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()