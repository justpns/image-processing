# import the necessary packages
import cv2

# used to access local directory
import os

# used to plot our images
import matplotlib.pyplot as plt

# used to change image size
from pylab import rcParams

import numpy as np

import imutils

def rect_to_bounding_box(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords


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
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

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
    print("Faces:\n", facefeatures)

	# # loop over the face detections
    for (i, x, y, w, h) in facefeatures:
        cropped_gray_face = gray_image[y:y + h, x:x + w]
        cropped_original_face = image[y:y + h, x:x + w]

		# show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# and draw them on the image
        for (x, y) in cropped_gray_face:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    
    cv2.imshow('Detected Faces', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()