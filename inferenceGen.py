from scipy.stats.stats import mode
import tensorflow.keras as keras
from tensorflow.python.keras import activations
import metrics as met
import cv2
import os
import numpy as np
from tensorflow import config
from sklearn.metrics import roc_auc_score , roc_curve
import pandas as pd
from keras import backend as K
from pathlib import *

gpus = config.experimental.list_physical_devices('GPU') 
config.experimental.set_memory_growth(gpus[0], True)

batch_size = 32


test_bilder = list()

for folder in os.listdir("data//UFC//testing//frames"):
    path = os.path.join("data//UFC//testing//frames",folder)
    #bildmappar.append(folder)
    for img in os.listdir(path):
        bild = os.path.join(path,img)
        test_bilder.append(bild)


test_etiketter = list()
path = "data//UFC//testing"
testnings_ettiketter = (x for x in Path(path).iterdir() if x.is_file())
for ettiket in testnings_ettiketter:
    test_etiketter.append(np.load(ettiket))

test_etiketter = np.concatenate(test_etiketter,axis=0)

def test_generator():
    while 1:
        for i in range(0,len(test_bilder),batch_size):
            batch_samples = test_bilder[i:i+batch_size]
            batch_labels = test_etiketter[i:i+batch_size]
            x_input = []
            y_output = []
            for j in range(len(batch_samples)):
                bild = cv2.imread(batch_samples[j])
                #bild = cv2.resize(bild,(360,240))
                x_input.append(bild)
                y_output.append(batch_labels[j])
            x_input = np.array(x_input)
            #x_input = x_input.reshape(x_input.shape[0],x_input.shape[1],x_input.shape[2],x_input.shape[3],1)
            x_input = x_input.astype('float32') / 255
            y_output = np.array(y_output)
            yield x_input, y_output


validation_steps = len(test_bilder) / batch_size
testGenObject = test_generator()
reconstructed_model = keras.models.load_model("modelUFC1.h5")

reconstructed_model.evaluate(testGenObject,steps=validation_steps,verbose=1)


y_score = reconstructed_model.predict(testGenObject,steps=validation_steps,verbose=1)

auc = roc_auc_score(test_etiketter,y_score=y_score)

print('AUC: ', auc*100, '%')