


""" 
from sklearn import metrics, model_selection
from sklearn import metrics
from tensorflow.keras.layers import Dense,Conv3D,MaxPooling3D,BatchNormalization,Flatten,Input, Add
from tensorflow.keras.models import Model , Sequential
from tensorflow import config
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling3D
import tensorflow.keras as keras

gpus = config.experimental.list_physical_devices('GPU') 
config.experimental.set_memory_growth(gpus[0], True)

model = Sequential()

model.add(Conv3D(input_shape =(240, 360, 3, 1),activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(MaxPooling3D(pool_size=(3,3,1),strides=(2,2,1)))
model.add(BatchNormalization())

model.add(Conv3D(activation="relu",filters=128,kernel_size=3,padding="same"))
model.add(MaxPooling3D(pool_size=(2,2,1)))

model.add(Conv3D(activation="relu",filters=256,kernel_size=2,padding="same"))
#model.add(MaxPooling3D(pool_size=(2,2,1)))
#model.add(GlobalAveragePooling3D())
model.add(Dense(24,activation="relu"))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
#metrics = [keras.metrics.AUC(),keras.metrics.binary_accuracy(),keras.metrics.Precision()]
#model.compile(metrics=[keras.metrics.AUC,keras.metrics.binary_accuracy,keras.metrics.Precision,],loss=keras.losses.binary_crossentropy)
model.compile(metrics=["binary_accuracy","AUC","Precision"],loss=keras.losses.BinaryCrossentropy)
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model,show_shapes=True)

input = Input((240,360,3,1))

x = Conv3D(64,3,padding="same")(input)
x = MaxPooling3D(pool_size=(3,3,3))(x)
x = Flatten()(x)
x = Dense(128)(x)

y = Dense(1)(input)
y = Flatten()(y)
y = Dense(128)(y)
y = Dense(128)(y)

x = Add()([x,y])
x = Dense(10)(x)
x = Dense(1)(x)

model = Model(inputs = input,outputs = x)
model.compile()
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model,show_shapes=True)


model = Sequential()


model.add(Dense(1,input_shape =(240, 360, 3, 1)))
#model.add(Dense(1))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(10))
model.add(Dense(1))

model.compile()
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model,show_shapes=True)

import cv2 
import numpy as np 
import os
from pathlib import *

path = "data/UFC"

films = list()
files = (x for x in Path(path).iterdir() if x.is_file())
for file in files:
    namn = str(file.name).split("_")[0]
    if(namn == "Normal"):
        namn = str(file.name).split("_")[0:2]
        namn = namn[0] + "_" + namn[1]
    #print(str(file.name).split("_")[0], "is a file!")
    films.append(namn)

with open('data//UCFCrime2Local//UCFCrime2Local//Test_split_AD.txt') as f:
    lines = f.readlines()

for x in range(len(lines)):
    hittade = False
    for j in range(len(lines)):
        if hittade is False and lines[x].strip() == films[j]:
            #print(lines[x])
            hittade = True
            break
    if hittade is False:
        print(lines[x])

print("hittade inget fel") 


import os
path = "data//UFC//training//frames"
#"dataset\adoc\Normal\01"
folders = os.listdir(path)
for folder in folders:
    files = os.listdir(os.path.join(path,folder))
    for file in files:
        name = int(file.split('.')[0])
        
        cur = os.path.join(path,folder,file)
        if (os.path.exists(cur)):
            name = str("%05d" % name)
            #dest = "../Datasets/Adoc/videos/Day1/testing/part3/" + str(name) + ".jpg"
            dest = os.path.join(path,folder,name +".jpg",)
            os.rename(cur,dest)
"""

