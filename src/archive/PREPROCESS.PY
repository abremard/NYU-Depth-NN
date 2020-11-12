# import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# rgbimg.paste(img)
# rgbimg.save('foo.jpg')

baseURL = "F:/NYU Data/nyu_depth_v2_raw"

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
# for idx, subfolder in enumerate(subfolders):
#     print("Scanning "+subfolder+"...")
#     for path in Path(subfolder).rglob('*.ppm'):
#         tmp = path.name.split(".")
#         if int(tmp[0].split("-")[1]) != imgID:
#             # print(imgID, imgTimestamp)
#             tmpRGBDict[nbRGB] = {"id": imgID, "RGBTimestamp": imgTimestamp, "RGBPath": RGBPath}
#             nbRGB = nbRGB + 1
#         imgID = int(tmp[0].split("-")[1])
#         imgTimestamp = int(tmp[1].split("-")[0])
#         RGBPath = path
#     for path in Path(subfolder).rglob('*.pgm'):
#         tmp = path.name.split(".")
#         if int(tmp[0].split("-")[1]) != imgID:
#             # print(imgID, imgTimestamp)
#             tmpDepthDict[nbDepth] = {"id": imgID, "DepthTimestamp": imgTimestamp, "DepthPath": depthPath}
#             nbDepth = nbDepth + 1
#         imgID = int(tmp[0].split("-")[1])
#         imgTimestamp = int(tmp[1].split("-")[0])
#         depthPath = str(path).replace('\\','/')

# tmpDF = pd.DataFrame.from_dict(tmpRGBDict, "index")
# RGBDF = RGBDF.append(tmpDF)
# tmpDF = pd.DataFrame.from_dict(tmpDepthDict, "index")
# DepthDF = DepthDF.append(tmpDF)

# JoinDF = pd.merge(DepthDF, RGBDF, on='id')
# JoinDF = JoinDF.iloc[1:]

# JoinDF.to_csv("JoinDF.csv")

JoinDF = pd.read_csv("JoinDF.csv")
# JoinDF = shuffle(JoinDF)

dataSize = 9000
trainSize = 8000

train = np.zeros((dataSize, 60, 80, 3), dtype=np.int)
train_label = np.zeros((dataSize, 60, 80, 1), dtype=np.int)

index = 0

for cursor, row in JoinDF.iterrows():
    imRGB = Image.open(row["RGBPath"].replace("D:", "F:"))
    test = imRGB
    imRGB = imRGB.resize((80, 60), Image.ANTIALIAS)
    imDepth = cv2.imread(row["DepthPath"].replace("D:", "F:"),cv2.IMREAD_GRAYSCALE)
    try:
        imDepth = cv2.resize(imDepth, dsize=(80, 60), interpolation=cv2.INTER_CUBIC)
        # if index == 0:
        #     fig = plt.figure()
        #     a = fig.add_subplot(2, 2, 1)
        #     plt.imshow(imDepth)
        #     a.set_title('Depth')
        #     a = fig.add_subplot(2, 2, 2)
        #     plt.imshow(imRGB)
        #     a.set_title('RGB')
        #     plt.show()
        imDepth = imDepth.reshape((60,80,1))
        if imRGB is None or imDepth is None or imRGB.size == () or imDepth.size == ():
            continue
        train[index] = np.array(imRGB)
        train_label[index] = imDepth
        index = index + 1
    except:
        break

    if index == dataSize - 1:
        break


print(np.shape(train), np.shape(train_label))

# MODEL
# UTILITY FUNCTIONS

def add_conv(mod, kernel, filters, strides = 1):
    return mod.add(keras.layers.Conv2D(filters= filters, kernel_size= kernel, strides= strides))

def add_batch_norm(mod):
    return mod.add(keras.layers.BatchNormalization())

def add_max_pooling(mod, kernel, strides = 1):
    return mod.add(keras.layers.MaxPooling2D(pool_size=kernel, strides = strides))

def add_relu(mod):
    return mod.add(keras.layers.ReLu())

def add_projection(mod,entry,exit):
    left = mod.add()

    right = mod.add()
    return 

x_train = train[:trainSize]
y_train = train_label[:trainSize]
x_test = train[trainSize+1:]
y_test = train_label[trainSize+1:]

# BUILD
# We're not using Sequential as it prevents res net skip connection
# https://stackoverflow.com/questions/42384602/implementing-skip-connections-in-keras
# model = keras.layers.Input(shape = (480,640,3))
# model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(keras.layers.MaxPooling2D((2, 2), padding='same'))
# model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(keras.layers.MaxPooling2D((2, 2), padding='same'))
# model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(keras.layers.UpSampling2D((2, 2), padding='same'))
# model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(keras.layers.UpSampling2D((2, 2), padding='same'))
# model.add(keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

input_img = keras.Input((60, 80, 3))

x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
decoded = keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)

model = keras.Model(input_img, decoded)

model.summary()

opt = keras.optimizers.Adam(learning_rate=0.00001)

model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

# x_train = x_train.reshape(-1, 480, 640, 3)
# y_train = y_train.reshape(-1, 480, 640, 3)
# x_test = x_test.reshape(-1, 480, 640, 1)
# y_test = y_test.reshape(-1, 480, 640, 1)

print(np.shape(x_train))
print(np.shape(y_train))

history = model.fit(x_train, y_train,
                epochs=100,
                batch_size=10,
                shuffle=True,
                validation_data=(x_test, y_test),
                )

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

for i in range(20):
    imageIndex = i
    trueImage = y_train[imageIndex]
    trueImage = trueImage[:, :, 0]
    modelImage = model.predict(x_train[imageIndex:imageIndex+1])
    modelImage = modelImage[0]
    modelImage = modelImage[:, :, 0]

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    plt.imshow(trueImage)
    a.set_title('trueImage')
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(modelImage)
    a.set_title('modelImage')
    plt.show()

# fig = plt.figure()
# a = fig.add_subplot(2, 2, 1)
# plt.imshow(y_train[0][:, :, 0])
# a.set_title('Depth')
# a = fig.add_subplot(2, 2, 2)
# plt.imshow(x_train[0])
# a.set_title('RGB')
# plt.show()


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