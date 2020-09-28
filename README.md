# Smile-Detection-ML-Vs-DL-Case-Study
This project was created as a case study to demonstrate how sometimes powerful deep learning methods are overkill to certain problems and can infact perform worse than machine learning techniques.

While exploring through Dlib's facial landmark detection model, I realized that I could use the facial landmarks to easily detect smiles in images and video streams. So to compare how this effective method compared to deep learning methods, I trained a smile detector model using the SMILES dataset by Daniel Hromada. The model obtained 93% accuracy and was put to test against the Dlib's landmark based model. To my surprise the Dlib's model actually performed better. I have mentioned all the details below.

## Package Requirements
For Dlib's ML based model:-
- Python3
- Numpy
- Dlib
- OpenCV (with HighGUI)
- Imutils

For Deep Learning based model:-
- Python3
- Tensorflow 2
- Numpy
- OpenCV (with HighGUI)
- Dlib
- Imutils

## How to run
For Dlib's ML based model:-

use the command from the terminal- <b>python/python3 detect_smile_ml.py -l shape_predictor_68_face_landmarks.dat</b>

<b>NOTE- the shape_predictor is the trained facial landmark model</b>

For Deep Learning based model:-

use the command from the terminal- <b>python/python3 detect_smile_dl.py -m smiles_model.hdf5</b>

<b> NOTE- the smiles_model.hdf5 is the model I trained on the SMILES dataset.</b>

<b>NOTE that I have added an optional argument to add a path to a video file to detect smiles. If that argument is not used, the script will default to using feed from web cam. The optional argument can be accesed as -v [path to video file]. For more information check out the python files</b>



