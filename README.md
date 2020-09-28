# Smile-Detection-ML-Vs-DL-Case-Study
This project was created as a case study to demonstrate how sometimes powerful deep learning methods are overkill to certain problems and can infact perform worse than machine learning techniques.

While exploring through Dlib's facial landmark detection model, I realized that I could use the facial landmarks to easily detect smiles in images and video streams. So to compare how this effective method ranked to deep learning methods, I trained a smile detector model using the SMILES dataset by Daniel Hromada. The model obtained 93% accuracy and was put to test against the Dlib's landmark based model. To my surprise the Dlib's model actually performed better. I have mentioned all the details below.

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

Running the above commands opens up a OpenCV highGUI window where the smiles are detected and labels are displayed. Checkout the output images for a preview.

I have also included the training script for training the smiles model. The dataset can be downloaded from <a href="https://github.com/meng1994412/Smile_Detection/tree/master/dataset" target='_blank'>this repo</a>.

I have also added a combined model running script that will run both the model on separate windows. The arguments are the same as the last two scripts. Run the command <b>python/python3 combined_model.py -l [path to dlib's landmark model] -m [path to trained smiles model]</b>
  
## Methodology
To detect the smiles, first detecting the face is necessary. To do this I used the Dlib's frontal face detector in both the models. Also I have improved the speed of detection by deliberately skipping some frames while detecting the face. This is done because during any sort of prediction from models, the face detection system creates the bottleneck which slows down the entire pipeline. Ofcourse this is only done for web cam faces with the assumption that the face will constantly be in front of the camera and skipping a few frames will not matter, but the same cannot be done for videos.

These are the steps I followed to achieve the results

For Dlib's ML model:-

1) Detected the landmark points and isolated the points on the sides of the lip and the jaw

2) Using experimentation, determined a threshold for the ratio of the distances between the lip points and the jaw points. If the ratio is above that threshold then it's highly likely that the person is smiling.

3) The distance was the Euclidean distance between the points


For Deep Learning model:-
1) Trained a tinyVGG network on the SMILES dataset.

2) Extarcted the face ROIs from the video stream and predicted it using the model.

## Inferences
Since the deep learning model is highly dependent of the dataset it was trained on, it cannot recognize smiles which do not display the teeth. Also the face has to be orientated in a general direction for it to perform better. On the other hand since the facial landmark detector works in almost any orientation and lighting conditions, the Euclidean distance ration between the points can easily be calculated. Hence in this simple application using deep learning is actually giving relatively poor results than machine learning. While we can improve the deep learning model by adding more data of different people and using augmentation techniques, the method would still be tedious and not worth the effort when a simple all pervasive model can be used.

Using this case study I want to convey that it is worthwhile to actually invest time looking at simple techniques rather than always going for deep learning based model which take time and resources to train. While deep learning is a very powerful and pervasive techniques for almost all the problems, it is not necessary to use it each time as sometimes for simpler applications machine learning techniques may work much better. 

Consult the output images and a output video present in the output file for more information.

## Output

Notice how the ML model uses the facial landmarks to calculate whether the person is smiling or not. Also notice how the DL model fails when the face is at various angels.

Watch the full video <a href='https://youtu.be/O8N49c_6wt4' target='_blank'>here</a>

<img src='output/output 1.png' width= 500 px>
<img src='output/output 2.png' width= 500 px>
<img src='output/output 3.png' width= 500 px>
<img src='output/output 4.png' width= 500 px>
