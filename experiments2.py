import os
from sklearn.model_selection import train_test_split
from pathlib import *
import numpy as np

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

val_X_1, val_X_2, val_Y1, val_Y_2 = train_test_split(test_bilder,test_etiketter,train_size=0.0001,random_state=100)

val_X_1 = val_X_1.append(val_X_2)

print("jeh")

OpticalFlow
SlowFast Network

transfer learning