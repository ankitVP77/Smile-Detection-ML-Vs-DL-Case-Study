from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import imutils
import argparse
import dlib

ap=argparse.ArgumentParser()
ap.add_argument('-m','--model',required=True,help='Path to saved dl model')
ap.add_argument('-v','--video',help='Path to optional video file')
args= vars(ap.parse_args())

# Using the dlib's face detector
detector= dlib.get_frontal_face_detector()
model= load_model(args['model'])

# If video file is not given use default web camera to read video input from.
v_flag=0
c_flag=0
if not args.get("video", False):
    cap= cv2.VideoCapture(0)
    c_flag=1
else:
    cap= cv2.VideoCapture(args['video'])
    v_flag=1

# Define variables to skip frames for face detection.
skip_f= 5
f_count=0
while(cap.isOpened()):
    time= cv2.getTickCount()
    ret, frame= cap.read()
    if ret==True:
        framec= imutils.resize(frame,height=360)
        framec= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the face on every 5th frame
        # NOTE THAT THIS SPEED UP METHOD IS ONLY USED FOR READING FRAMES FROM WEB-CAM
        if c_flag==1:
            if f_count%skip_f==0:
                faces= detector(framec)
        else:
            faces= detector(framec)

        if faces:
            for f in faces:
                # Get ROI of the detected face
                roif= framec[f.top():f.bottom(), f.left():f.right()]
                # Preprocess the face ROI to pass into trained model
                roif= cv2.resize(roif, (28,28))
                roif= roif.astype('float')/255.0
                roif= img_to_array(roif)
                roif= np.expand_dims(roif, axis=0)

                # Get predictions from model and assign a label
                (no_smile, smile)= model.predict(roif)[0]
                if no_smile > smile:
                    label= 'Not Smiling'
                else:
                    label= 'Smiling'
                # Put bounding-box and labels on the frame
                cv2.rectangle(frame, (f.left(),f.top()), (f.right(), f.bottom()),
                          (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(frame, label, (f.left(), f.top()-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2, cv2.LINE_AA)

        else:
            cv2.putText(frame, 'Something is Wrong!!', (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Calculate and display frame along with FPS count
        fps = (cv2.getTickCount() - time) / cv2.getTickFrequency()
        cv2.putText(frame, 'FPS=> {:.1f}'.format(fps*1000), (10, 20), cv2.FONT_HERSHEY_COMPLEX,
                    0.7, (0, 127, 0), 2, cv2.LINE_AA)
        cv2.imshow('Smile Detector', frame)
        f_count+=1
        k= cv2.waitKey(1)
        if k==ord('q'):
            break

    else:
        break

cv2.destroyAllWindows()
cap.release()



