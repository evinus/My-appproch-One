


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



"""

import os
import numpy as np
from pathlib import *

etiketter = list()
path = "data//UFC//testing"
träningsEttiketer = (x for x in Path(path).iterdir() if x.is_file())
for ettiket in träningsEttiketer:
    etiketter.append(np.load(ettiket))

i = 0
j = 0

bilder = list()
bildmappar = list()
for folder in os.listdir("data//UFC//testing//frames"):
    path = os.path.join("data//UFC//testing//frames",folder)
    bildmappar.append(folder)
    for img in os.listdir(path):
        bild = os.path.join(path,img)
        bilder.append(bild)
        i += 1
    if i != len(etiketter[j]):
        print(str("Name:%s i = %i, etiketter = %i" % (bildmappar[j],i,  len(etiketter[j]) )))
        
    j += 1
    i = 0
    


etiketter = np.concatenate(etiketter,axis=0)

if len(bilder) != len(etiketter):
    print("något gått fel")



model.add(keras.layers.Conv2D(input_shape =(240, 320, 3),activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(activation="relu",filters=128,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=128,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=128,kernel_size=3,padding="same"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(activation="relu",filters=256,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=256,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=256,kernel_size=3,padding="same"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(activation="relu",filters=512,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=512,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=512,kernel_size=3,padding="same"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(activation="relu",filters=512,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=512,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=512,kernel_size=3,padding="same"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(128,activation="relu"))
#model.add(keras.layers.GlobalAveragePooling3D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1028,activation="relu"))
model.add(keras.layers.Dense(64,activation="relu"))
#model.add(keras.layers.Dense(10,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))


import os
path = "data//UFC//testing//frames"
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



def data_generator():
    while 1:
        for i in range(0,len(X_train),batch_size):
            batch_samples = X_train[i:i+batch_size]
            batch_labels = Y_train[i:i+batch_size]
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
        #del batch_labels,batch_samples,x_input,y_output
#var1, var2 = data_generator()
#labels = np.reshape(labels,labels.shape[1])

genObject = data_generator()
testGenObject = test_generator()

""" for i in range(len(X_test)):
    X_test[i] = cv2.imread(X_test[i])
    X_test[i] = cv2.resize(X_test[i],(360 ,240))
X_test = np.array(X_test)
#X_test =  X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],X_test.shape[3],1)
X_test = X_test.astype('float32') / 255 """

model.add(keras.layers.Conv2D(input_shape =(240, 320, 3),activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Conv2D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.Dense(10,activation="relu"))
model.add(keras.layers.Dense(1,activation="relu"))
#model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256,activation="relu"))
model.add(keras.layers.Dense(64,activation="relu"))
#model.add(keras.layers.Dense(10,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))