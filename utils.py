

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


    def on_epoch_end(self):
        gc.collect()
        np.random.shuffle(self.indices)
