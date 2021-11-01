

from tensorflow.keras.utils import Sequence
import math
import numpy as np
import cv2
import gc
#from skimage.io import imread
import tensorflow as tf
from keras import backend

class Dataloader(Sequence):
    def __init__(self,x_set,y_set,batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(len(self.x))

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)



    def createStacks(self,batch_x,batch_y,j):
        train_X = list()
        train_y = list()
        
        for i in range(self.batch_size- j):
            train_X.append(batch_x[i:i + self.batch_size])
            train_y.append(batch_y[i + self.batch_size])
        for k in range(j):
            train_X.append(batch_x[len(batch_x)-j])
            train_y.append(batch_y[len(batch_y)-j])
        train_X = np.array(train_X)
        train_X = train_X.reshape(train_X.shape[0],train_X.shape[2],train_X.shape[3],train_X.shape[4],train_X.shape[1])
        train_X = train_X.astype("float32") / 255
        return train_X,np.array(train_y)



    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        #batch_x = self.x[inds]
        #batch_y = self.y[inds]
        batch_x = [self.x[index] for index in inds]
        batch_y = [self.y[index] for index in inds]
        batch_x = np.array([cv2.imread(file_name)
            for file_name in batch_x])
        batch_x = batch_x.astype("float32") / 255
        
        batch_x = batch_x.reshape(batch_x.shape[0],batch_x.shape[1],batch_x.shape[2],batch_x.shape[3],1)
        batch_y = np.array(batch_y) 
        return batch_x, batch_y


    def on_epoch_end(self) -> None:
        #gc.collect()
        np.random.shuffle(self.indices)
