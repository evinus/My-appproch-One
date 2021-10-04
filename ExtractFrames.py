import cv2 
import numpy as np 
import os
from pathlib import *

path = "data/UFC"

films = list()
files = (x for x in Path(path).iterdir() if x.is_file())
for file in files:
    print(str(file.name).split(".")[0], "is a file!")
    films.append(file)




for film in films:

    cap = cv2.VideoCapture(str(film))
    if(cap.isOpened()):
        print("Öppnades")
    else:
        cap.open()
    len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    mapp = str(film.name).split(".")[0]
    while(cap.isOpened):
        ret, frame = cap.read()
        name = "data/UFC/testing/frames/%s/%d.jpg" % (mapp, i)
        if not os.path.isdir("data/UFC/testing/frames/%s" % mapp):
            os.mkdir("data/UFC/testing/frames/%s" % mapp)
        if(ret and not os.path.isfile(name)):
            frame = cv2.resize(frame,(320,240))
            cv2.imwrite(name,frame)
        if(i == len_frames):
            break
        i += 1

    cap.release()

