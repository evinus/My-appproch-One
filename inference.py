
import tensorflow.keras as keras
#from tensorflow.python.keras.saving.save import load_model
#from tensorflow.python.keras import callbacks
import metrics as met
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import config
from utils import Dataloader
from sklearn.metrics import roc_auc_score , roc_curve
from pathlib import *

gpus = config.experimental.list_physical_devices('GPU') 
config.experimental.set_memory_growth(gpus[0], True)

test_bilder = list()

#for folder in os.listdir("data//ped2//testing//frames"):
#    path = os.path.join("data//ped2//testing//frames",folder)

#    for img in os.listdir(path):
#        bild = os.path.join(path,img)
#        test_bilder.append(bild)

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

#test_etiketter = list()
#path = "data//ped2//testing"
#testnings_ettiketter = (x for x in Path(path).iterdir() if x.is_file())
#for ettiket in testnings_ettiketter:
#    test_etiketter.append(np.load(ettiket))
#test_etiketter = np.load("data//frame_labels_ped2_2.npy")




batch_size = 16
#test_etiketter = np.concatenate(test_etiketter,axis=0)
#with open("frame_labels_UCFLocal","wb") as f:
    #np.save(f,test_etiketter)

test_gen = Dataloader(test_bilder,test_etiketter,batch_size)

#model = keras.models.load_model("model4.h5",custom_objects={"f1_m": met.f1_m,"precision_m": met.precision_m,"recall_m":met.recall_m})
reconstructed_model = keras.models.load_model("modelUFC3D_4-ep004-loss0.367-val_loss0.421.tf")
#model.compile(optimizer="adam",metrics=["acc",met.f1_m,met.precision_m,met.recall_m],loss="binary_crossentropy")
#reconstructed_model.evaluate(test_gen)
#print(results)

y_score = reconstructed_model.predict(test_gen,verbose=1)


auc = roc_auc_score(test_etiketter,y_score=y_score)
print('AUC: ', auc*100, '%')