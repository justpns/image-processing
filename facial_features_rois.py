# import the necessary packages
import cv2

# used for accessing url to download files
import urllib.request as urlreq

# used to access local directory
import os

# used to plot our images
import matplotlib.pyplot as plt
import numpy as np

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

eyeDet = cv2.CascadeClassifier("./haar_models/haarcascade_eye.xml")
open_eye_Det = cv2.CascadeClassifier("./haar_models/haarcascade_eye_tree_eyeglasses.xml")
left_eyeDet = cv2.CascadeClassifier("./haar_models/haarcascade_lefteye_2splits.xml")
right_eyeDet = cv2.CascadeClassifier("./haar_models/haarcascade_righteye_2splits.xml")

mouthDet = cv2.CascadeClassifier("./haar_models/haarcascade_mcs_mouth.xml")
mouth_smilingDet = cv2.CascadeClassifier("./haar_models/haarcascade_smile.xml")

# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "lbfmodel.yaml"
LBFmodel_file = "data/" + LBFmodel

# check if file is in working directory
if (LBFmodel in os.listdir(os.curdir)):
    print("File exists")
else:
    # download picture from url and save locally as lbfmodel.yaml, < 54MB
    urlreq.urlretrieve(LBFmodel_url, LBFmodel)
    print("File downloaded")

# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

cv2.namedWindow("preview")
capture = cv2.VideoCapture(0)

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

    if len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        facefeatures = faces

    # Print coordinates of detected faces
    # print("Faces:\n", facefeatures)

    faceCount = 1
    for (x, y, w, h) in facefeatures:
        cropped_gray_face = gray_image[y:y + h, x:x + w]
        cropped_original_face = image[y:y + h, x:x + w]

        # show the face number
        cv2.putText(image, "Face #{}".format(faceCount), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.rectangle(image,(x, y),(x + w,y + h),(0, 255, 0), 2)
        ++faceCount

        # Eyes detection
        # check first if eyes are open (with glasses taking into account)

        eyes = eyeDet.detectMultiScale(cropped_gray_face, 1.3, 3)

        open_eyes_glasses = open_eye_Det.detectMultiScale(
            cropped_gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        if len(open_eyes_glasses) == 2:
            for (ex, ey, ew, eh) in open_eyes_glasses:
                cv2.rectangle(cropped_original_face, (ex,ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        else:
            first_half_face = frame[y:y + h, x + int(w/2):x + w]
            first_half_gray_face = cropped_original_face[y:y + h, x + int(w/2):x + w]

            second_half_face = frame[y:y + h, x:x + int(w/2)]
            second_half_gray_face = cropped_original_face[y:y + h, x:x + int(w/2)]

            # Detect the left eye
            left_eye = left_eyeDet.detectMultiScale(
                first_half_gray_face,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            # Detect the right eye
            right_eye = right_eyeDet.detectMultiScale(
                second_half_gray_face,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

        # Mouth detection
        # check first if mouth is smiling
        
        smiling_mouth = mouth_smilingDet.detectMultiScale(
            cropped_gray_face,
            scaleFactor=1.7,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(smiling_mouth) == 2:
            for (mx, my, mw, mh) in smiling_mouth:
                color = (0,0,255)
                cv2.rectangle(cropped_original_face, (mx, my), (mx + mw, my + mh), color, 2)
        else:
            
            mouths = mouthDet.detectMultiScale(
                cropped_gray_face,
                scaleFactor=1.7,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
            
            for (mx, my, mw, mh) in mouths:
                color = (0,0,255)
                cv2.rectangle(image, (mx, my), (mx + mw, my + mh), color, 2)
        
         # Detect landmarks on "image_gray"
        _, landmarks = landmark_detector.fit(gray_image, np.array(facefeatures))

        for landmark in landmarks:
            for x,y in landmark[0]:
                # display landmarks on "image_cropped"
                # with white colour in BGR and thickness 1
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow('Face Features Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()