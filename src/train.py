"""
    TODO :
    - make the project research-friendly :
        - build report : e.g. plot 20 images with 3 variants (RGB, ground truth, NNDepth) on the same page 
        - keep track of hyperparams, metadata, configs, date, results (include in report)
    - only start improving model when the 2 above are done...
"""

from tensorflow import keras
from sklearn.utils import shuffle
from random import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def fetch_images(filepath, disk_name, data_size, train_size, shuffle, width, height):
    """ read images on disk using RGB-depth mapping dataframe, downsize them and split them into training and testing set

    Args:
        filepath (str): mapping dataframe path
        disk_name (str): "F:"...
        data_size (int):
        train_size (int):
        shuffle (boolean): if dataframe is suffled before split
        width (int): downsized image width
        height (int): downsized image height

    Returns:
        x_train, y_train, x_test, y_test : numpy array objects
    """

    JoinDF = pd.read_csv(filepath)
    
    if shuffle:
        JoinDF = shuffle(JoinDF)

    data = np.zeros((data_size, width, height, 3), dtype=np.int)
    data_label = np.zeros((data_size, width, height, 1), dtype=np.int)

    index = 0

    for cursor, row in JoinDF.iterrows():
        imRGB = Image.open(row["RGBPath"].replace("D:", disk_name))
        test = imRGB
        imRGB = imRGB.resize((height, width), Image.ANTIALIAS)
        imDepth = cv2.imread(row["DepthPath"].replace("D:", disk_name),cv2.IMREAD_GRAYSCALE)
        try:
            imDepth = cv2.resize(imDepth, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
            imDepth = imDepth.reshape((width,height,1))
            if imRGB is None or imDepth is None or imRGB.size == () or imDepth.size == ():
                continue
            data[index] = np.array(imRGB)
            data_label[index] = imDepth
            index = index + 1
        except:
            break

        if index == data_size - 1:
            break

    # print(np.shape(data), np.shape(data_label))

    # DATA
    x_train = data[:train_size]
    y_train = data_label[:train_size]
    x_test = data[train_size+1:]
    y_test = data_label[train_size+1:]

    print(np.shape(x_train))
    print(np.shape(y_train))
    print(np.shape(x_test))
    print(np.shape(y_test))

    return x_train, y_train, x_test, y_test

def auto_encoder_v1(width, height):
    """ Autoencoder version 1.0 : 2 down-convolution layers + 2 up-convolutional layers

    Args:
        width (int): image width
        height (int): image height

    Returns:
        model: keras model
    """ 

    # MODEL
    input_img = keras.Input((width, height, 3))
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

    return model

def train(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    """ Train any given keras model

    Args:
        model (keras.model): model to train
        x_train (np.array): train data
        y_train (np.array): train label
        x_test (np.array): test data
        y_test (np.array): test label
        epochs (int):
        batch_size (int):

    Returns:
        history: model training history, useful for plotting
    """

    # TRAIN
    history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, y_test),
                    )

    return history

def plot(history):
    """ Plot training history of a model

    Args:
        history (keras.model.train.history):
    """    

    # PLOT
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

def predict(model, x_train, y_train, nb_images):
    """ Predict the depth of 2D images once model is trained and plot RGB and prediction side by side
        TODO:
            - add ground truth to each unit
            - generate report

    Args:
        model (keras.model): 
        x_train (np.array): 
        y_train (np.array): 
        nb_images (int): number of images to show on report
    """

    # PREDICTION
    for i in range(nb_images):
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

def test():
    """
    
    """
    x_train, y_train, x_test, y_test = fetch_images("JoinDF.csv", "F:", 9000, 8000, False, 60, 80)
    model = auto_encoder_v1(60, 80)
    history = train(model, x_train, y_train, x_test, y_test, 100, 10)
    plot(history)
    predict(model, x_train, y_train, 20)