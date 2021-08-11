import tensorflow.keras as keras
#from tensorflow.python.keras import callbacks
import metrics as met
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import config
from sklearn.preprocessing import MinMaxScaler

gpus = config.experimental.list_physical_devices('GPU') 
config.experimental.set_memory_growth(gpus[0], True)

bilder = list()
for folder in os.listdir("data\ped2//testing//frames"):
    path = os.path.join("data\ped2//testing//frames",folder)
    for img in os.listdir(path):
        bild = os.path.join(path,img)
        bilder.append(cv2.imread(bild))

bilder = np.array(bilder)

bilder = bilder.reshape(bilder.shape[0],bilder.shape[1],bilder.shape[2],bilder.shape[3],1)

scaler = MinMaxScaler()
bilder = scaler.fit_transform(bilder)

labels = np.load("data/frame_labels_ped2.npy")

labels = np.reshape(labels,2010)

X_train, X_test, Y_train, Y_test = train_test_split(bilder,labels,test_size=0.2)


batch_size = 16
model = keras.Sequential()

model.add(keras.layers.Conv3D(input_shape =(240, 360, 3, 1),activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.MaxPooling3D(pool_size=(2,2,1)))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv3D(activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(keras.layers.MaxPooling3D(pool_size=(2,2,1)))

model.add(keras.layers.Conv3D(activation="relu",filters=128,kernel_size=3,padding="same"))
model.add(keras.layers.MaxPooling3D(pool_size=(2,2,1)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dense(64,activation="relu"))
model.add(keras.layers.Dense(10,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))

metrics = [keras.metrics.categorical_crossentropy,keras.metrics.Accuracy,keras.metrics.Precision,met.f1_m]

model.compile(optimizer="adam",metrics=["acc",met.f1_m,met.precision_m,met.recall_m],loss="binary_crossentropy")
#model.summary()
filepath = filepath = 'model2-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')]

model.fit(X_train,Y_train,batch_size=batch_size,verbose=1,epochs=50,callbacks=callbacks,validation_split=0.1)

model.evaluate(X_test,Y_test,batch_size=batch_size)
