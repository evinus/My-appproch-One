
import tensorflow.keras as keras
#from tensorflow.python.keras.saving.save import load_model
#from tensorflow.python.keras import callbacks
import metrics as met
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import config

from sklearn.metrics import roc_auc_score , roc_curve

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

bilder = bilder.astype('float32') / 255

labels = np.load("data/frame_labels_ped2_2.npy")

#labels = np.reshape(labels,labels.shape[1])

X_train, X_test, Y_train, Y_test = train_test_split(bilder,labels,test_size=0.2, random_state= 100)


batch_size = 64
#model = keras.models.load_model("model4.h5",custom_objects={"f1_m": met.f1_m,"precision_m": met.precision_m,"recall_m":met.recall_m})
reconstructed_model = keras.models.load_model("model5.h5")
#model.compile(optimizer="adam",metrics=["acc",met.f1_m,met.precision_m,met.recall_m],loss="binary_crossentropy")
reconstructed_model.evaluate(X_test,Y_test,batch_size=batch_size)
#print(results)
y_score = reconstructed_model.predict(X_test,batch_size=batch_size)

auc = roc_auc_score(Y_test,y_score=y_score)
print('AUC: ', auc*100, '%')