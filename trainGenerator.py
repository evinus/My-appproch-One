
from scipy.stats.stats import mode
import tensorflow.keras as keras
from tensorflow.python.keras import activations
#from tensorflow.python.keras import callbacks
import metrics as met
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import config
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score , roc_curve
import pandas as pd
from keras import backend as K
from pathlib import *

gpus = config.experimental.list_physical_devices('GPU') 
config.experimental.set_memory_growth(gpus[0], True)

bilder = list()
bildmappar = list()
for folder in os.listdir("data//UFC//training//frames"):
    path = os.path.join("data//UFC//training//frames",folder)
    bildmappar.append(folder)
    for img in os.listdir(path):
        bild = os.path.join(path,img)
        bilder.append(bild)

etiketter = list()
path = "data//UFC//training"
träningsEttiketer = (x for x in Path(path).iterdir() if x.is_file())
for ettiket in träningsEttiketer:
    etiketter.append(np.load(ettiket))

etiketter = np.concatenate(etiketter,axis=0)
#labels = np.load("data/frame_labels_avenue.npy")
#labels = np.reshape(labels,labels.shape[1])
#X_train, X_test, Y_train, Y_test = train_test_split(bilder,labels,test_size=0.2, random_state= 100)

""" for i in range(len(X_test)):
    X_test[i] = cv2.imread(X_test[i])
    X_test[i] = cv2.resize(X_test[i],(360 ,240))
X_test = np.array(X_test)
#X_test =  X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],X_test.shape[3],1)
X_test = X_test.astype('float32') / 255 """

batch_size = 32

def data_generator():
    while 1:
        for i in range(0,len(bilder),batch_size):
            batch_samples = bilder[i:i+batch_size]
            batch_labels = etiketter[i:i+batch_size]
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


#var1, var2 = data_generator()
#labels = np.reshape(labels,labels.shape[1])


""" bildData = list()
for bild in X_test:
    bildData.append(cv2.imread(bild))

bildData = np.array(bildData)
X_test = bildData """

#nb_train_samples = len(Y_train)


""" df = pd.DataFrame(data={"x_col":X_train,"y_col":Y_train})#columns=(["x_col","y_col"]))
df["y_col"] = df["y_col"].astype(str)


dataget = ImageDataGenerator(rescale=1. / 255)
train_get = dataget.flow_from_dataframe(dataframe=df,x_col="x_col",y_col="y_col",class_mode="binary",target_size=(240,360),batch_size=batch_size,color_mode="rgb")

if K.image_data_format() == 'channels_first':
    input_shape = (3,240, 360)
else:
    input_shape = (240, 360, 3) """

model = keras.Sequential()

model.add(keras.layers.Conv2D(input_shape =(240, 320, 3),activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(activation="relu",filters=128,kernel_size=3,padding="same"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(activation="relu",filters=128,kernel_size=2,padding="same"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dense(64,activation="relu"))
#model.add(keras.layers.GlobalAveragePooling3D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256,activation="relu"))
model.add(keras.layers.Dense(64,activation="relu"))
model.add(keras.layers.Dense(10,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))

metrics = [keras.metrics.categorical_crossentropy,keras.metrics.Accuracy,keras.metrics.Precision,met.f1_m]
steps = len(bilder) / batch_size
#model.compile(optimizer="adam",metrics=["acc",met.f1_m,met.precision_m,met.recall_m],loss="binary_crossentropy")
model.compile(optimizer="adam",metrics=["binary_accuracy","AUC","Precision","Recall"],loss="binary_crossentropy")
model.summary()
filepath = 'model2-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")]#,keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')]
genObject = data_generator()
model.fit(genObject,verbose=1,epochs=1,steps_per_epoch=steps,callbacks=callbacks)#,validation_data=(X_test,Y_test))

model.save("modelUFC1.h5")
#reconstructed_model = keras.models.load_model("modelGen2.h5")

#np.testing.assert_allclose(model.predict(X_test), reconstructed_model.predict(X_test))
#np.testing.assert_allclose(model.evaluate(X_test,Y_test,batch_size=batch_size), reconstructed_model.evaluate(X_test,Y_test,batch_size=batch_size))
#model.evaluate(X_test,Y_test,batch_size=batch_size)


#y_score = model.predict(X_test,batch_size=batch_size)

#auc = roc_auc_score(Y_test,y_score=y_score)

#print('AUC: ', auc*100, '%')