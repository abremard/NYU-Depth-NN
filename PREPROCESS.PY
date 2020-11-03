# import tensorflow as tf
# from tensorflow import keras
from random import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import os
import pandas as pd

import cv2
from PIL import Image

# rgbimg.paste(img)
# rgbimg.save('foo.jpg')

baseURL = "../NYU Data/living_rooms_part1"

subfolders = [ f.path for f in os.scandir(baseURL) if f.is_dir() ]

nbRGB = 0
imgID = 0
imgTimestamp = 0

RGBDF = pd.DataFrame()
DepthDF = pd.DataFrame()

tmpDict = {}

# RGB files
for subfolder in subfolders:
    for path in Path(subfolder).rglob('*.ppm'):
        tmp = path.name.split(".")
        if int(tmp[0].split("-")[1]) != imgID:
            # print(imgID, imgTimestamp)
            tmpDict[nbRGB] = {"id": imgID, "RGBPath": path}
            nbRGB = nbRGB + 1
        imgID = int(tmp[0].split("-")[1])
        imgTimestamp = int(tmp[1].split("-")[0])

tmpDF = pd.DataFrame.from_dict(tmpDict, "index")
RGBDF = RGBDF.append(tmpDF)

print(RGBDF)

nbDepth = 0
imgID = 0
imgTimestamp = 0

tmpDict = {}

# Depth files
for subfolder in subfolders:
    for path in Path(subfolder).rglob('*.pgm'):
        tmp = path.name.split(".")
        if int(tmp[0].split("-")[1]) != imgID:
            # print(imgID, imgTimestamp)
            tmpDict[nbDepth] = {"id": imgID, "DepthPath": path}
            nbDepth = nbDepth + 1
        imgID = int(tmp[0].split("-")[1])
        imgTimestamp = int(tmp[1].split("-")[0])

tmpDF = pd.DataFrame.from_dict(tmpDict, "index")
DepthDF = DepthDF.append(tmpDF)

print(DepthDF)

JoinDF = pd.merge(DepthDF, RGBDF, on='id')

print(JoinDF)

# print(nbDepth)

# print(nbRGB, nbDepth)

# im1 = mpimg.imread('../NYU Data/living_rooms_part1/living_room_0001a/r-1294634097.072535-123122978.ppm')
# im2 = mpimg.imread('../NYU Data/living_rooms_part1/living_room_0001a/r-1294634097.213239-133124416.ppm')
# im3 = mpimg.imread('../NYU Data/living_rooms_part1/living_room_0001a/r-1294634097.278244-135124703.ppm')
# im4 = mpimg.imread('../NYU Data/living_rooms_part1/living_room_0001a/r-1294634097.455731-147126428.ppm')

# fig = plt.figure()
# a = fig.add_subplot(2, 2, 1)
# plt.imshow(im1)
# a.set_title('Depth')
# a = fig.add_subplot(2, 2, 2)
# plt.imshow(im2)
# a.set_title('RGB')
# a = fig.add_subplot(2, 2, 3)
# plt.imshow(im3)
# a.set_title('Depth')
# a = fig.add_subplot(2, 2, 4)
# plt.imshow(im4)
# a.set_title('RGB')

# im5 = cv2.imread('../NYU Data/living_rooms_part1/living_room_0001a/d-1294634097.267685-134627249.pgm',-1)
# im6 = cv2.imread('../NYU Data/living_rooms_part1/living_room_0001a/d-1294634097.344780-136629404.pgm',-1)
# im7 = cv2.imread('../NYU Data/living_rooms_part1/living_room_0001a/d-1294634097.415183-144638024.pgm',-1)
# im8 = cv2.imread('../NYU Data/living_rooms_part1/living_room_0001a/d-1294634097.444114-146640179.pgm',-1)

# fig = plt.figure()
# a = fig.add_subplot(2, 2, 1)
# plt.imshow(im5)
# a.set_title('Depth')
# a = fig.add_subplot(2, 2, 2)
# plt.imshow(im6)
# a.set_title('RGB')
# a = fig.add_subplot(2, 2, 3)
# plt.imshow(im7)
# a.set_title('Depth')
# a = fig.add_subplot(2, 2, 4)
# plt.imshow(im8)
# a.set_title('RGB')

# plt.show()