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

