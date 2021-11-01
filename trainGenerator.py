
#from scipy.stats.stats import mode
import tensorflow as tf
import math
import tensorflow.keras as keras
#from tensorflow.python.keras import activations
from tensorflow.python.keras import callbacks
#import metrics as met
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import config
#from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score , roc_curve
#import pandas as pd
from keras import backend as K
from pathlib import *
import pickle
from tensorflow import device
from utils import Dataloader
from tensorflow.keras.utils import plot_model


import gc
#gc.enable()
#gc.set_debug(gc.DEBUG_LEAK)
#tf.compat.v1.disable_eager_execution()

gpus = config.experimental.list_physical_devices('GPU') 
config.experimental.set_memory_growth(gpus[0], True)

def LoadTrainTestData():
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
    #X_train, X_test, Y_train, Y_test = train_test_split(bilder,etiketter,test_size=0.0001, random_state= 100)

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
    return bilder,etiketter,test_bilder,test_etiketter






def CreateModel():
    model = keras.Sequential()

    model.add(keras.layers.Conv3D(input_shape =(240, 320, 3, 1),activation="relu",filters=64,kernel_size=3,padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv3D(activation="relu",filters=64,kernel_size=3,padding="same"))
    model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Conv3D(activation="relu",filters=64,kernel_size=3,padding="same"))
    model.add(keras.layers.MaxPooling3D(pool_size=(2,2,1)))

    model.add(keras.layers.Conv3D(activation="relu",filters=128,kernel_size=3,padding="same"))
    #model.add(keras.layers.Conv3D(activation="relu",filters=128,kernel_size=3,padding="same"))
    #model.add(keras.layers.Conv3D(activation="relu",filters=128,kernel_size=3,padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling3D(pool_size=(2,2,1)))

    model.add(keras.layers.Conv3D(activation="relu",filters=256,kernel_size=3,padding="same"))
    #model.add(keras.layers.Conv3D(activation="relu",filters=256,kernel_size=3,padding="same"))
    #model.add(keras.layers.Conv3D(activation="relu",filters=256,kernel_size=3,padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling3D(pool_size=(2,2,1)))

    model.add(keras.layers.Conv3D(activation="relu",filters=512,kernel_size=3,padding="same"))
    #model.add(keras.layers.Conv3D(activation="relu",filters=512,kernel_size=3,padding="same"))
    #model.add(keras.layers.Conv3D(activation="relu",filters=512,kernel_size=3,padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling3D(pool_size=(2,2,1)))

    model.add(keras.layers.Conv3D(activation="relu",filters=512,kernel_size=3,padding="same"))
    #model.add(keras.layers.Conv3D(activation="relu",filters=512,kernel_size=3,padding="same"))
    #model.add(keras.layers.Conv3D(activation="relu",filters=512,kernel_size=3,padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling3D(pool_size=(2,2,1)))

    model.add(keras.layers.Dense(128,activation="relu"))
    #model.add(keras.layers.GlobalAveragePooling3D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1028,activation="relu"))
    model.add(keras.layers.Dense(64,activation="relu"))
    #model.add(keras.layers.Dense(10,activation="relu"))
    model.add(keras.layers.Dense(1,activation="sigmoid"))
    return model



#metrics = [keras.metrics.categorical_crossentropy,keras.metrics.Accuracy,keras.metrics.Precision,met.f1_m]



if __name__ == "__main__":
    
    batch_size = 16
    model = CreateModel()
    #model = keras.models.load_model("modelUFC3D-ep001-loss0.482.h5-val_loss0.502.h5")
    bilder, etiketter, test_bilder, test_etiketter = LoadTrainTestData()
    
    train_gen = Dataloader(bilder,etiketter,batch_size)

    test_gen = Dataloader(test_bilder,test_etiketter,batch_size)

    train_steps = math.ceil( len(bilder) / batch_size)
    validation_steps =math.ceil( len(test_bilder) / batch_size)

    #model.compile(optimizer="adam",metrics=["acc",met.f1_m,met.precision_m,met.recall_m],loss="binary_crossentropy")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),metrics=["binary_accuracy","AUC","Precision","Recall","TruePositives","TrueNegatives","FalsePositives","FalseNegatives"],loss="binary_crossentropy")
    model.summary()
    plot_model(model,show_shapes=True,show_layer_names=False)
    quit()
    filepath = 'modelUFC3D_2-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.tf'
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min"),keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')]

    history = model.fit(train_gen,verbose=1,epochs=20,steps_per_epoch=train_steps,callbacks=callbacks,validation_data=test_gen,validation_steps=validation_steps)#,validation_data=(X_test,Y_test))
    model.save("modelUFC3D_1",save_format='tf')
    with open("history1.pk","wb") as handle:
        pickle.dump(history.history,handle)

    reconstructed_model = keras.models.load_model("modelUFC3D_1")

    np.testing.assert_allclose(model.predict(test_gen,steps=validation_steps), reconstructed_model.predict(test_gen,steps=validation_steps))
    np.testing.assert_allclose(model.evaluate(test_gen,steps=validation_steps), reconstructed_model.evaluate(test_gen,steps=validation_steps))
    model.evaluate(test_gen,steps=validation_steps,verbose=1)


    #y_score = model.predict(test_gen,steps=validation_steps)

    #auc = roc_auc_score(test_etiketter,y_score=y_score)

    #print('AUC: ', auc*100, '%')

