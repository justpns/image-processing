import os
import cv2
import numpy as np

faceDet = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("./models/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("./models/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("./models/haarcascade_frontalface_alt_tree.xml")

eyeDet = cv2.CascadeClassifier("./models/haarcascade_eye.xml")
open_eye_Det = cv2.CascadeClassifier("./models/haarcascade_eye_tree_eyeglasses.xml")
left_eyeDet = cv2.CascadeClassifier("./models/haarcascade_mcs_lefteye.xml")
right_eyeDet = cv2.CascadeClassifier("./models/haarcascade_mcs_righteye.xml")

mouthDet = cv2.CascadeClassifier("./models/haarcascade_mcs_mouth.xml")
mouth_smilingDet = cv2.CascadeClassifier("./models/haarcascade_smile.xml")

cv2.namedWindow("preview")
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    image = frame.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceDet.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

    face_two = faceDet_two.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

    face_three = faceDet_three.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

    face_four = faceDet_four.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
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
    print("Faces:\n", faces)
        
    for (x,y,w,h) in facefeatures:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0), 2)
        
        gray_face = gray[y:y+h, x:x+w]
        cropped_face = image[y:y+h, x:x+w]
        
        # Eyes detection
        # check first if eyes are open (with glasses taking into account)
        
        eyes = eyeDet.detectMultiScale(gray_face, 1.3, 3)
        
        open_eyes_glasses = open_eye_Det.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(open_eyes_glasses) == 2:
            for (ex, ey, ew, eh) in open_eyes_glasses:
                cv2.rectangle(cropped_face, (ex,ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        else:
            first_half_face = frame[y:y + h, x + int(w/2):x + w]
            first_half_gray_face = gray[y:y + h, x + int(w/2):x + w]

            second_half_face = frame[y:y + h, x:x + int(w/2)]
            second_half_gray_face = gray[y:y + h, x:x + int(w/2)]
            
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

            eye_status = '1' # we suppose the eyes are open

            # For each eye check wether the eye is closed.
            # If one is closed we conclude the eyes are closed
            for (ex,ey,ew,eh) in right_eye:
                color = (0,0,255)
                # pred = predict(right_face[ey:ey+eh,ex:ex+ew],model)
                # if pred == 'closed':
                #     eye_status='0'
                #     color = (0,0,255)
                cv2.rectangle(first_half_face,(ex,ey),(ex+ew,ey+eh),color, 2)
            for (ex,ey,ew,eh) in left_eye:
                color = (0,0,255)
                # pred = predict(left_face[ey:ey+eh,ex:ex+ew],model)
                # if pred == 'closed':
                #     eye_status='0'
                #     color = (0,0,255)
                cv2.rectangle(second_half_face,(ex,ey),(ex+ew,ey+eh),color, 2)
                
            # eyes_detected[name] += eye_status
        
        # Mouth detection
        # check first if mouth is smiling
        
        smiling_mouth = mouth_smilingDet.detectMultiScale(
            gray_face,
            scaleFactor=1.7,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(smiling_mouth) == 2:
            for (mx, my, mw, mh) in smiling_mouth:
                color = (0,0,255)
                cv2.rectangle(cropped_face, (mx, my), (mx + mw, my + mh), color, 2)
        else:
            
            mouths = mouthDet.detectMultiScale(
                gray_face,
                scaleFactor=1.7,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
            
            for (mx, my, mw, mh) in mouths:
                color = (0,0,255)
                cv2.rectangle(cropped_face, (mx, my), (mx + mw, my + mh), color, 2)
        
        
    cv2.imshow('Facial Features: Face, Eyes, Mouth', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()