from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sidekick.nn.conv.minivgg import MiniVgg
from sidekick.plot.plot_graph import plot_graph
import numpy as np
from imutils import paths
import imutils
import argparse
import cv2
import os

ap= argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to dataset')
ap.add_argument('-m', '--model', default=".\smiles_model.hdf5", help='Path to save trained model')
args= vars(ap.parse_args())

print('[NOTE]:- Loading Images...\n')
img_paths= list(paths.list_images(args['dataset']))

data=[]
labels=[]

e=0
for i,img_path in enumerate(img_paths):
    img= cv2.imread(img_path,0)
    img= imutils.resize(img,width=28,height=28)
    img= img_to_array(img)
    data.append(img)

    label= img_path.split(os.path.sep)[-3]
    if label=='positives':
        label= 'smiling'
    else:
        label= 'not_smiling'
    labels.append(label)
    e+=1
    if e%1000==0:
        print('Loaded-> {}/{}'.format(e,len(img_paths)))

data= np.array(data, dtype="float")/ 255.0

# Performing one-hot encoding using a combination of Label Encoder that creates
# a vector of classes and later use to_categorical of keras that creates the one-hot class matrix
lb= LabelEncoder().fit(labels)
labels= to_categorical(lb.transform(labels), 2)

# Since the SMILES dataset has a class imbalance I created a normalization for the weight updates
# Defining class weights
# Taking sum across the labels to get the totals for each class
classTotal= labels.sum(axis=0)
# Dividing the higher number and creating a factor by which the lower is updates
classWeights= classTotal.max()/ classTotal

(trainX, testX, trainY, testY)= train_test_split(data,labels,test_size=0.2,random_state=36)

print('[NOTE]:- Compiling model...\n')
model= MiniVgg.build(28, 28, 1, 2)
model.compile(loss='binary_crossentropy', metrics=['accuracy'],
              optimizer="adam")

print('[NOTE]:- Training model...\n')
# Class-weights are added here.
H= model.fit(trainX,trainY, validation_data=(testX,testY), batch_size= 64, epochs= 18, class_weight=classWeights)

model.save(args['model'])

print('[NOTE]:- Evaluating model...\n')
preds= model.predict(testX)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_))

print('[NOTE]:- Plotting graph...\n')
plot_graph(18,H,save=True)
