

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
        self.längd = len(self.x)
        self.antalPositiva = 0
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)



    def createStacks(self,batch_x,batch_y,j,index):
        train_X = list()
        train_y = list()
        sista = len(batch_x)
        if j == 0:
            for i in range(self.batch_size):
                train_X.append(batch_x[i:i + self.batch_size])
                train_y.append(batch_y[i + self.batch_size])
        else:
            for i in range(self.batch_size):
                if(i < (32 - j - 16)):
                    train_X.append(batch_x[i:i + self.batch_size])
                    train_y.append(batch_y[i + self.batch_size])
                else:
                    clip_x = list()
                    for k in range(self.batch_size):
                        clip_x.append(batch_x[sista - 1])
                    train_X.append(clip_x)
                    train_y.append(batch_y[sista - 1])
        train_X = np.array(train_X)
        #if train_X.shape != (16,16,240,320,3):
        #    print(train_X.shape)
        train_X = train_X.reshape(train_X.shape[0],train_X.shape[2],train_X.shape[3],train_X.shape[4],train_X.shape[1])
        train_X = train_X.astype("float32") / 255
        train_y = np.array(train_y)
        #if train_y.max == 1:
        #    print(1)
        return train_X, train_y

    def __getitem__(self, index):
        #längd =len(self) - 2
        #index = längd
        #längd2 =
        i = (index + 2) * self.batch_size
        #l = (längd2 + 2) * self.batch_size
        #k = längd * self.batch_size
        j = 0
        if i > (self.längd -2):
            j = i - len(self.x)
            #print(j)
            #print(index)
        batch_x = self.x[index * self.batch_size:(index + 2) * self.batch_size - j]
        batch_y = self.y[index * self.batch_size:(index + 2) * self.batch_size - j]
        batch_x = np.array([cv2.imread(file_name)
               for file_name in batch_x])
        batch_y = np.array(batch_y)
        #if(batch_y.max == 1):
        #    self.antalPositiva += 1
        batch_x,batch_y = self.createStacks(batch_x,batch_y,j,i)
        return batch_x, batch_y
   
   


    #def on_epoch_end(self) -> None:
        #gc.collect()
        #np.random.shuffle(self.indices)
