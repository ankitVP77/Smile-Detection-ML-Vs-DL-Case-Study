from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import argparse
import dlib

ap= argparse.ArgumentParser()
ap.add_argument('-l', '--landmark_model', required=True,
                help='Path to dlib trained 68 pts landmark model')
ap.add_argument('-m','--model',required=True,help='Path to saved model')
ap.add_argument('-v', '--video', help='Optional path to detect smiles in video files')

args= vars(ap.parse_args())

face_detector= dlib.get_frontal_face_detector()
shape_predictor= dlib.shape_predictor(args['landmark_model'])
model= load_model(args['model'])

def calcdist(pt1,pt2):
    return np.sqrt(((pt2[0]-pt1[0])**2)+((pt2[1]-pt1[1])**2))

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

if not args.get("video",False):
    cap= cv2.VideoCapture(0)
else:
    cap= cv2.VideoCapture(args['video'])

num_f=0
skip_f=5
label_ml= None
label_dl= None
cv2.namedWindow('Smile Detection ML',cv2.WINDOW_NORMAL)
cv2.namedWindow('Smile Detection DL',cv2.WINDOW_NORMAL)
while(cap.isOpened()):
    time= cv2.getTickCount()
    ret, frame= cap.read()
    frame_ml= frame.copy()
    frame_dl= frame.copy()
    framec = cv2.cvtColor(frame_dl, cv2.COLOR_BGR2GRAY)
    if ret==True:
        if num_f%skip_f==0:
            faces= face_detector(frame)

        if faces:
            for f in faces:
                roif = framec[f.top():f.bottom(), f.left():f.right()]
                roif = cv2.resize(roif, (28, 28))
                roif = roif.astype('float') / 255.0
                roif = img_to_array(roif)
                roif = np.expand_dims(roif, axis=0)

                ratio, points = getlipjawratio(frame_ml, f)

                (no_smile, smile) = model.predict(roif)[0]
                if no_smile > smile:
                    label_dl = 'Not Smiling'
                else:
                    label_dl = 'Smiling'
                cv2.rectangle(frame_dl, (f.left(), f.top()), (f.right(), f.bottom()),
                              (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_dl, label_dl, (f.left(), f.top() - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)

                if ratio and points:
                    if ratio > 0.85:
                        label_ml= 'Smiling'
                    else:
                        label_ml= 'Not Smiling'
                    for p in points:
                        cv2.circle(frame_ml,p,2,(255,0,0),2,cv2.LINE_AA)
                    cv2.putText(frame_ml, label_ml, (points[0][0]-115,points[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0,255,0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame_ml, 'Something is Wrong!!', (10,45), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0,0,255), 2, cv2.LINE_AA)

        else:
            cv2.putText(frame_dl, 'Something is Wrong!!', (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_ml, 'Something is Wrong!!', (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

        fps = (cv2.getTickCount() - time) / cv2.getTickFrequency()
        cv2.putText(frame_ml, 'FPS=> {:.1f}'.format(fps * 1000), (10, 20), cv2.FONT_HERSHEY_COMPLEX,
                    0.7, (0, 127, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_dl, 'FPS=> {:.1f}'.format(fps * 1000), (10, 20), cv2.FONT_HERSHEY_COMPLEX,
                    0.7, (0, 127, 0), 2, cv2.LINE_AA)
        cv2.imshow('Smile Detection DL', frame_dl)
        cv2.imshow('Smile Detection ML', frame_ml)
        num_f += 1
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()