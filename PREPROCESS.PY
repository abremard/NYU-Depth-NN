# import tensorflow as tf
from tensorflow import keras
from random import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import os
import sys
import pandas as pd

import cv2
from PIL import Image

# rgbimg.paste(img)
# rgbimg.save('foo.jpg')

baseURL = "D:/NYU Data/nyu_depth_v2_raw"

subfolders = [ f.path for f in os.scandir(baseURL) if f.is_dir() ]

nbRGB = 0
imgID = 0
RGBPath = ""
imgTimestamp = 0
nbDepth = 0
imgID = 0
imgTimestamp = 0
depthPath = ""

RGBDF = pd.DataFrame()
DepthDF = pd.DataFrame()

tmpRGBDict = {}
tmpDepthDict = {}

# RGB files
for idx, subfolder in enumerate(subfolders):
    print("Scanning "+subfolder+"...")
    for path in Path(subfolder).rglob('*.ppm'):
        tmp = path.name.split(".")
        if int(tmp[0].split("-")[1]) != imgID:
            # print(imgID, imgTimestamp)
            tmpRGBDict[nbRGB] = {"id": imgID, "RGBTimestamp": imgTimestamp, "RGBPath": RGBPath}
            nbRGB = nbRGB + 1
        imgID = int(tmp[0].split("-")[1])
        imgTimestamp = int(tmp[1].split("-")[0])
        RGBPath = path
    for path in Path(subfolder).rglob('*.pgm'):
        tmp = path.name.split(".")
        if int(tmp[0].split("-")[1]) != imgID:
            # print(imgID, imgTimestamp)
            tmpDepthDict[nbDepth] = {"id": imgID, "DepthTimestamp": imgTimestamp, "DepthPath": depthPath}
            nbDepth = nbDepth + 1
        imgID = int(tmp[0].split("-")[1])
        imgTimestamp = int(tmp[1].split("-")[0])
        depthPath = str(path).replace('\\','/')

tmpDF = pd.DataFrame.from_dict(tmpRGBDict, "index")
RGBDF = RGBDF.append(tmpDF)
tmpDF = pd.DataFrame.from_dict(tmpDepthDict, "index")
DepthDF = DepthDF.append(tmpDF)

JoinDF = pd.merge(DepthDF, RGBDF, on='id')
JoinDF = JoinDF.iloc[1:]

JoinDF.to_csv("JoinDF.csv")

train = []
train_label = []

for index, row in JoinDF.iterrows():
    imRGB = mpimg.imread(row["RGBPath"])
    imDepth = cv2.imread(row["DepthPath"],-1)
    if imRGB is None or imDepth is None or imRGB.shape == () or imDepth.shape == ():
        continue
    train.append(imRGB)
    train_label.append(imDepth)
    # fig = plt.figure()
    # a = fig.add_subplot(2, 2, 1)
    # plt.imshow(imDepth)
    # a.set_title('Depth')
    # a = fig.add_subplot(2, 2, 2)
    # plt.imshow(imRGB)
    # a.set_title('RGB')
    # print(imDepth.shape, imRGB.shape)
    # plt.show()
    if index == 2000:
        break

print(np.shape(train), np.shape(train_label))

# MODEL
# UTILITY FUNCTIONS

def add_batch_norm(mod):
    return mod.add(keras.layers.BatchNormalization())

def add_conv(mod, kernel, filters, strides = 1):
    return mod.add(keras.layers.Conv2D(filters= filters, kernel_size= kernel, strides= strides))

def add_max_pooling(mod, kernel, strides = 1):
    return mod.add(keras.layers.MaxPooling2D(pool_size=kernel, strides = strides))

def add_relu(mod):
    return mod.add(keras.layers.ReLu())

def add_projection(mod,entry,exit):
    left = mod.add()

    right = mod.add()
    return 

# BUILD
# We're not using Sequential as it prevents res net skip connection
# https://stackoverflow.com/questions/42384602/implementing-skip-connections-in-keras
model = keras.layers.Input(shape(480,640,3))
model.add(
    keras.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
    )
)







# print(JoinDF)

# JoinDF.to_csv('JoinDF.csv')

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