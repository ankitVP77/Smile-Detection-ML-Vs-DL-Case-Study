import cv2
import dlib
import numpy as np
import argparse
import imutils

ap= argparse.ArgumentParser()
ap.add_argument('-l', '--landmark_model', required=True,
                help='Path to dlib trained 68 pts landmark model')
ap.add_argument('-v', '--video', help='Optional path to detect smiles in video files')

args= vars(ap.parse_args())

face_detector= dlib.get_frontal_face_detector()
shape_predictor= dlib.shape_predictor(args['landmark_model'])

# Function to calculate the euclidean distance between points
def calcdist(pt1,pt2):
    return np.sqrt(((pt2[0]-pt1[0])**2)+((pt2[1]-pt1[1])**2))

# Function to calculate ratio of distances between lips and jaw
def getlipjawratio(img, face):
    if face:
        landmark=shape_predictor(img, face)
        points=[]
        points.append((landmark.part(48).x,landmark.part(48).y))
        points.append((landmark.part(54).x,landmark.part(54).y))
        points.append((landmark.part(6).x,landmark.part(6).y))
        points.append((landmark.part(10).x,landmark.part(10).y))
        lipdist=calcdist(points[0],points[1])
        jawdist=calcdist(points[2],points[3])
        return lipdist/jawdist, points
    else:
        return False, False

# If video file is not given use default web camera to read video input from.
v_flag=0
c_flag=0
if not args.get("video",False):
    cap= cv2.VideoCapture(0)
    c_flag=1
else:
    cap= cv2.VideoCapture(args['video'])
    v_flag=1

# Define variables to skip frames for face detection.
num_f=0
skip_f=5
label= None
while(cap.isOpened()):
    time= cv2.getTickCount()
    ret, frame= cap.read()
    frame= imutils.resize(frame, height=360)
    if ret==True:
        # Detect the face on every 5th frame
        # NOTE THAT THIS SPEED UP METHOD IS ONLY USED FOR READING FRAMES FROM WEB-CAM
        if c_flag==1:
            if num_f%skip_f==0:
                faces= face_detector(frame)
        else:
            faces = face_detector(frame)
        if faces:
            for f in faces:
                # Get the lip to jaw distance ratio
                ratio, points= getlipjawratio(frame, f)
                if ratio and points:
                    # If the ratio is above a certain threshold the person is smiling
                    # THIS THRESHOLD IS A HYPERPARAMETER AND HAS BEEN CALCULATED THROUGH EXPERIMENTATION
                    if ratio > 0.85:
                        label= 'Smiling'
                    else:
                        label= 'Not_Smiling'
                    for p in points:
                        # Draw the landmark points on the face
                        cv2.circle(frame,p,2,(255,0,0),2,cv2.LINE_AA)
                    cv2.putText(frame, label, (points[0][0]-115,points[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0,255,0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'Something is Wrong!!', (10,45), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0,0,255), 2, cv2.LINE_AA)

        else:
            cv2.putText(frame, 'Something is Wrong!!', (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
        # Calculate and display fps
        fps= ((cv2.getTickCount()- time)/cv2.getTickFrequency())*1000
        cv2.putText(frame, 'FPS= {:.2f}'.format(fps), (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,127,0), 2, cv2.LINE_AA)
        cv2.imshow('Smile Detection', frame)
        num_f+=1
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    else:
        break

cv2.destroyAllWindows()
cap.release()

