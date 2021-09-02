########################################################################################################################
# saves activations by layer
# project: articulation angle detection
# auth.:   gyenist
# date:    2020.02.20
# log:
#
########################################################################################################################

import os
from pathlib import Path
import cv2
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import preprocess as pp
import modelBuilder as mb
import tensorflow as tf
import keras.backend as bck
import pickle
import matplotlib.image as mpimg
import numpy as np
from keras import models
import pandas as pd

import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()

MODEL_NAME = "c4"
WEIGHTS_NAME = "c41"
RESCALE_SHAPE = (200, 400, 3)  # (600, 1400, 3)
RESCALE_SIZE = (RESCALE_SHAPE[0], RESCALE_SHAPE[1])
BATCH_SIZE = 1

projectPath = "C:/Dev/Projects/art_angle"
dataPath = "C:/Dev/Projects/art_angle/data"
framesPath = "C:/Dev/Projects/art_angle/data/frames"
modelPath = projectPath + "/models/" + MODEL_NAME + ".h5"
weightPath = projectPath + "/models/" + WEIGHTS_NAME + ".hdf5"
framesPathRescale = framesPath + "/rescale_" + str(RESCALE_SHAPE[0]) + "_" + str(RESCALE_SHAPE[1])
testDataOutPath = projectPath + "/test_data" + "/" + WEIGHTS_NAME

if not os.path.exists(testDataOutPath +"_activations_2nd_conv"):
    os.makedirs(testDataOutPath +"_activations_2nc_conv")

dfTest = pp.loadLabels(dataPath + "/_1140test.txt")
dfOnePic = pp.loadLabels(dataPath + "/_1140oneline.txt")


def aa_diff(y_true, y_pred):
    return bck.mean(bck.abs(y_pred - y_true)) * 87.01


def mse(y_true, y_pred):
    return bck.mean((y_pred - y_true) ** 2)


def aa_perc_err(y_true, y_pred):
    return 100*abs(y_true-y_pred)/y_true


def mean(list):
    return sum(list)/len(list)



model = mb.buildModel(MODEL_NAME, RESCALE_SHAPE[0], RESCALE_SHAPE[1], RESCALE_SHAPE[2])
model.compile(loss="mean_squared_error", optimizer="adam")
model.load_weights(weightPath)
print("running forward prop with predict_generator...")
predDatagen = ImageDataGenerator(rescale=1. / 255, fill_mode="nearest")

predGenerator = predDatagen.flow_from_dataframe(dataframe=dfOnePic,
                                                directory=framesPathRescale,
                                                x_col="frameID",
                                                y_col="angle",
                                                class_mode="other",
                                                batch_size=BATCH_SIZE,
                                                target_size=RESCALE_SIZE,
                                                shuffle=False)

layer_outputs = [layer.output for layer in model.layers[:12]]

activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict_generator(predGenerator, verbose=1)

first_layer_activation = activations[4]
print(first_layer_activation.shape)


for i in range(0, first_layer_activation.shape[3]):
    fig = plt.figure()
    plt.matshow(first_layer_activation[0, :, :, i], cmap='viridis')
    plt.savefig(testDataOutPath +"_activations_2nd_conv" + "/" + str(i) + ".png")
    plt.close(fig)
