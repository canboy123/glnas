import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from keras.datasets import cifar10
from keras.utils import to_categorical
import os
import numpy as np
import logging
import sys
import coloredlogs
import time
from datetime import date, datetime
from sklearn.model_selection import train_test_split

from glnasmodel import Glnas

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from operations import *
from model import createStartingGLNASModel, addLayerToGLNASModel, addClassificationLayerToGLNASModel, Trainer, \
    oneOutputLayerAfterFlatten, createStemForGLNASModel
from dataset import datasetBuilder

CUDA_DEVICE_INDEX = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE_INDEX
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

########################################
#            INITIALIZATION            #
########################################
epochs = 1
batch_size = 128
num_classes = 10
input_shape = (32, 32, 3)
flatten_shape = 32 * 32 * 3
inputs = tf.keras.Input(shape=input_shape)
num_of_layer_feature_extraction = 20
num_of_layer_classification = 0

isTf = True
# isTf = False

useBlock = True
# useBlock = False

useDataAug = True
# useDataAug = False

useImgGenerator = True
useImgGenerator = False

useLayerRestriction = True
useLayerRestriction = False

useResnet = True
# useResnet = False

skipPreResnet = True
# skipPreResnet = False

useKernelReg = True
# useKernelReg = False

layer_b4_flat = "average"
layer_b4_flat = "global"

use_split_train = True
# use_split_train = False

freeze_pre_layer = True
freeze_pre_layer = False

eval_performance = "logit"
eval_performance = "accuracy"

top_k = 1
top_k = 2
top_k = 4
# top_k = 5

merge_or_add = "merge"
# merge_or_add = "add"

use_only_add_layer = True
use_only_add_layer = False

relu_position = "last"
relu_position = "first"

use_preprocessing_layer = True
# use_preprocessing_layer = False

use_weight_decay = True
# use_weight_decay = False

bool_save_model = True
# bool_save_model = False

bool_load_model = True
bool_load_model = False

early_stop = True
early_stop = False

### NO LONGER USE ###
use_separated_loss = True
# use_separated_loss = False

repeat = 1

split_val_data = 0.5
stddev_score = 0.05

score_gamma = 0.7

decay_steps = int(epochs * 50000/ batch_size)
weight_decay = 0.0
if use_weight_decay:
    weight_decay = 3e-4

# dd/mm/YY
if bool_load_model is not True:
    d1 = datetime.now().strftime("%Y%m%d%H00_"+CUDA_DEVICE_INDEX)
else:
    d1 = "202305171500_0"

CONST_SAVED_MODEL_OUTDIR = f"../saved_models/cifar10_tf_e{epochs}_{d1}"
CONST_SAVED_MODEL_DIR = CONST_SAVED_MODEL_OUTDIR+"/"+"saved_model_dir"
Path(f"{CONST_SAVED_MODEL_DIR}").mkdir(parents=True, exist_ok=True)

load_model_layer_index = 13
CONST_LOAD_MODEL_PATH = f"{CONST_SAVED_MODEL_OUTDIR}/saved_model_dir_{load_model_layer_index}"

today = date.today()

logname = f"{CONST_SAVED_MODEL_OUTDIR}/cifar10_log_{d1}.log"

modelSummaryStr = ""
CONST_OPTIMIZER = "sgd"
# CONST_OPTIMIZER = "adam"

learning_rate = False
learning_rate = 0.01

CONST_SPLIT_FILTER_SIZE = 3 # Split the size of the filter into 32, 64, 128, 256, 512
CONST_STARTING_FILTER = 32

config = {
    "use_layer_restriction": useLayerRestriction,
    "use_resnet_mode": useResnet,
    "skip_preresnet": skipPreResnet,
    "relu_position": relu_position,
}

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

CONST_MODEL_LOSS = 'categorical_crossentropy'
CONST_MODEL_METRICS = ['accuracy']

########################################
#               LOGGING                #
########################################
logger = logging.getLogger(logname)
coloredlogs.install(logger=logger)
logger.propagate = False

coloredFormatter = coloredlogs.ColoredFormatter(
    fmt='%(asctime)s %(funcName)s L%(lineno)-3d %(message)s',
    level_styles=dict(
        debug=dict(color='white'),
        info=dict(color='blue'),
        warning=dict(color='yellow', bright=True),
        error=dict(color='red', bold=True, bright=True),
        critical=dict(color='black', bold=True, background='red'),
    ),
    field_styles=dict(
        name=dict(color='white'),
        asctime=dict(color='red'),
        funcName=dict(color='blue'),
        lineno=dict(color='white'),
    )
)


logger.setLevel(level=logging.DEBUG)
logStreamFormatter = logging.Formatter(
  fmt=f"%(levelname)-8s %(asctime)s \t %(filename)s @function %(funcName)s line %(lineno)s - %(message)s",
  datefmt="%H:%M:%S"
)
consoleHandler = logging.StreamHandler(stream=sys.stdout)
consoleHandler.setFormatter(fmt=coloredFormatter)
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(hdlr=consoleHandler)
logger.setLevel(level=logging.DEBUG)

logFileFormatter = logging.Formatter(
    fmt=f"%(levelname)s %(asctime)s \t %(funcName)s L%(lineno)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
fileHandler = logging.FileHandler(filename=logname)
fileHandler.setFormatter(logFileFormatter)
fileHandler.setLevel(level=logging.INFO)

logger.addHandler(fileHandler)

########################################
#          DATASET FUNCTION            #
########################################
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

def normalize_image(image, mean, std):
    for channel in range(3):
        image[:,:,channel] = (image[:,:,channel] - mean[channel]) / std[channel]
    return image

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    train_norm = normalize_image(np.array(train_norm),
                                 mean=CIFAR_MEAN,
                                 std=CIFAR_STD)
    test_norm = normalize_image(np.array(test_norm),
                                mean=CIFAR_MEAN,
                                std=CIFAR_STD)

    # return normalized images
    return train_norm, test_norm

def split_train_data(X_train, Y_train):
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=split_val_data, shuffle=True)

    return X_train, X_val, Y_train, Y_val

def changeToTensorDataset(x_data, y_data, batch_size):
    # Prepare the training dataset.
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(len(x_data)).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

########################################
#          SAVE/LOAD FUNCTION          #
########################################
def save_model(model, layer_index):
    # Save the trained model
    Path(f"{CONST_SAVED_MODEL_DIR}_{layer_index}").mkdir(parents=True, exist_ok=True)
    print("SAVE MODEL PATH", f"{CONST_SAVED_MODEL_DIR}_{layer_index}")
    try:
        model.save(f"{CONST_SAVED_MODEL_DIR}_{layer_index}", save_format="tf")
    except Exception:
        print("exception saved model to h5 format")
        model.save(f"{CONST_SAVED_MODEL_DIR}_{layer_index}/saved_model.h5")

def load_saved_model(path):
    print("LOAD MODEL PATH", path)
    if path != "" and path is not None:
        model = load_model(f"{path}")

    return model

########################################
#          GET BEST FUNCTION           #
########################################
def getModelSummaryFromKeras(s):
    global modelSummaryStr
    modelSummaryStr += f"{s}\n"

def softmax_stable(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

def tanh(x):
    return np.tanh(x)

def writeToFile(scores):
    print("write scores", scores)
    file1 = open(f"{CONST_SAVED_MODEL_OUTDIR}/scores_{d1}.txt", "w")
    file1.writelines([f"{score}\n" for score in scores])
    file1.close()

def readFromFile():
    print(f"scores_{d1}.txt")
    file1 = open(f"{CONST_SAVED_MODEL_OUTDIR}/scores_{d1}.txt", "r")
    scores = file1.readlines()
    file1.close()
    scores = [float(score) for score in scores]
    print("read scores", scores)

    return scores

def lr_schedule(epoch, learning_rate):
    if epoch < 10:
        return learning_rate
    else:
        return learning_rate * tf.math.exp(-0.1)

def getOptimizer(cur_num_layer):
    if CONST_OPTIMIZER == "adam":
        optimizer = tf.keras.optimizers.Adam(decay=weight_decay)
    else:
        new_learning_rate = learning_rate/cur_num_layer
        print(f"new_learning_rate = {new_learning_rate}, cur_num_layer = {cur_num_layer}")

        if use_split_train:
            step_in_1_epoch = (50000 * (1.0 - split_val_data)) // batch_size
        else:
            step_in_1_epoch = 50000 // batch_size

        step_in_1_epoch = int(step_in_1_epoch)

        use_learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                        [step_in_1_epoch*2, step_in_1_epoch*85],   # In tensorflow, this is based on the step
                        [new_learning_rate, new_learning_rate / 10., new_learning_rate / 100.]
                    )
        decay_steps = int(epochs*50000/batch_size)
        use_learning_rate = tf.keras.experimental.CosineDecay(new_learning_rate, decay_steps=decay_steps)
        optimizer = tf.keras.optimizers.SGD(use_learning_rate, momentum=0.9)
        # optimizer = tf.keras.optimizers.SGD(new_learning_rate, momentum=0.9)

    return optimizer


########################################
#     ADDING LAYER MODEL FUNCTION      #
########################################
def trainFeatureExtractionLayers(trainX, trainY, valX=None, valY=None, testX=None, testY=None):
    print("Train Feature Extraction Layers........")
    global modelSummaryStr, bool_load_model
    isTrainFeatureDone = False
    modelSummaryStr = ""
    if useBlock:
        useSearchSpace = featureBlockSearchSpace
    else:
        useSearchSpace = featureSearchSpace

    # Used for the avgpooling layer before the flatten
    # Default was 32 because the size of the image is 32
    # The size should be reduced into half if it is going to do the down sample
    avg_pool_size = 32


    logger.info(msg=f"Feature Extraction Search Space: \n {useSearchSpace}")

    earlystop = False

    steps_per_epoch = None
    if useDataAug:
        if useImgGenerator:
            datagen = ImageDataGenerator(
                # featurewise_center=True,  # set input mean to 0 over the dataset
                # samplewise_center=False,  # set each sample mean to 0
                # featurewise_std_normalization=True,  # divide inputs by std of the dataset
                # samplewise_std_normalization=False,  # divide each input by its std
                # zca_whitening=False,  # apply ZCA whitening
                rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                # validation_split=0.2  # randomly flip images
                zoom_range=0.2  # randomly zoom
            )
            datagen.fit(trainX)
            trainDataset = datagen.flow(trainX, trainY, batch_size=batch_size)

            steps_per_epoch = trainX.shape[0] // batch_size
        else:
            builder = datasetBuilder()
            trainDataset = builder.build_dataset(
                trainY, trainX, batch_size, training=True, repeat=repeat)
    else:
        trainDataset = changeToTensorDataset(trainX, trainY, batch_size)

    fullDataset = trainDataset
    if valX is not None and valY is not None:
        valDataset = changeToTensorDataset(valX, valY, batch_size)
        fullDataset = valDataset
        # fullDatasetX = np.concatenate((trainX, valX), axis=0)
        # fullDatasetY = np.concatenate((trainY, valY), axis=0)
        # fullDataset = changeToTensorDataset(fullDatasetX, fullDatasetY, batch_size)

    glnas = Glnas(input_shape, num_classes, batch_size, num_of_layer_feature_extraction, featureBlockSearchSpace, classificationSearchSpace,
                  decay_steps, use_preprocessing_layer=use_preprocessing_layer, use_kernel_reg=useKernelReg, layer_b4_flat=layer_b4_flat, top_k=top_k, merge_or_add=merge_or_add,
                  use_only_add_layer=use_only_add_layer, ending_avg_pool_size=avg_pool_size,
                  config=config)


    scores = []
    max_score = 0
    optimizer = getOptimizer(cur_num_layer=1)
    # Load model
    if bool_load_model and os.path.isdir(CONST_LOAD_MODEL_PATH):
        logger.info(msg=f"Load model from: {CONST_LOAD_MODEL_PATH}")
        best_model = load_saved_model(CONST_LOAD_MODEL_PATH)
        print(f"============== Loaded Best Model ===============")
        best_model.summary()
        best_model.compile(optimizer=optimizer, loss=CONST_MODEL_LOSS, metrics=CONST_MODEL_METRICS)
        glnas.set_best_model(best_model)

        scores = readFromFile()

        # Continue to train for the next model
        layer_index = load_model_layer_index+1

        if layer_index > num_of_layer_feature_extraction:
            layer_index = load_model_layer_index
            isTrainFeatureDone = True

    else:
        bool_load_model = False
        logger.debug(msg=f"Skip from loading the model: {CONST_LOAD_MODEL_PATH}")
        glnas.createFirstNetwork(isTf, setFilter=CONST_STARTING_FILTER)
        optimizer = getOptimizer(cur_num_layer=1)
        glnas.start_training(trainDataset, fullDataset, epochs, 0, optimizer)

    multiplier = 1
    setFilter = CONST_STARTING_FILTER
    for i in range(num_of_layer_feature_extraction):
        down_sample = False
        if i >= 0  and i < num_of_layer_feature_extraction // CONST_SPLIT_FILTER_SIZE * multiplier:
            setFilter = setFilter
        else:
            setFilter = setFilter * 2
            multiplier = multiplier + 1
            down_sample = True
            avg_pool_size = int(avg_pool_size / 2)
            glnas.set_ending_avg_pool_size(avg_pool_size)

        if bool_load_model and i < load_model_layer_index:
            continue
        del optimizer
        layer_index = i + 1

        optimizer = getOptimizer(cur_num_layer=layer_index)
        # Create multiple models with the ops in the search space
        glnas.createMiddleNetwork(layer_index, setFilter=setFilter, isTf=isTf, down_sample=down_sample, freeze_pre_layer=freeze_pre_layer)

        # Training each model for the op in the search space
        acc, loss, logits = glnas.start_training(trainDataset, fullDataset, epochs, layer_index, optimizer)

        # Get the best model from multiple models
        top_index_logit, top_index_acc, best_index_acc, best_index_logit = glnas.getBestIndex(acc, loss, logits)

        # Get the score from the best logit to stop the training early if needed
        if eval_performance == "logit":
            score = logits[best_index_logit]
        else:
            score = acc[best_index_acc]
        scores.append(score)

        # Select and pick the only best model
        if eval_performance == "logit":
            glnas.selectBestModel(layer_index, top_index_logit)
        else:
            glnas.selectBestModel(layer_index, top_index_acc)

        # Return the best model and save it
        best_model = glnas.get_best_model()
        best_model.compile(optimizer=optimizer, loss=CONST_MODEL_LOSS, metrics=CONST_MODEL_METRICS)

        print(f"============== ({layer_index}) best_model model ===============")
        best_model.summary()
        if bool_save_model:
            save_model(best_model, layer_index)
            writeToFile(scores)

        del best_model
        logger.info(msg=f"Current Layer: {layer_index}")
        logger.info(msg=f"acc: {acc}\n loss: {loss}\n logit: {logits}")
        logger.info(
            msg=f"best index logit, acc [best score]: {best_index_logit}, {best_index_acc} [{logits[best_index_logit]}, "
                f"{acc[best_index_acc]}]")

        max_score = max(scores)
        if early_stop is True and len(scores) > 1:
            if scores[-1] < max_score - stddev_score:
                layer_index = i - 1
                logger.info(msg=f"Early stop searching at layer: {i}")
                earlystop = True
                break

    best_model = glnas.get_best_model()
    best_model.compile(optimizer=optimizer, loss=CONST_MODEL_LOSS, metrics=CONST_MODEL_METRICS)
    # Save the model as "final" model directory
    if bool_save_model:
        layer_index = "final"
        save_model(best_model, layer_index)

    print(f"============== final model ===============")
    best_model.summary(print_fn=getModelSummaryFromKeras)
    logger.info(msg=f"Final classification model summary:\n {modelSummaryStr}")

    out = best_model.evaluate(testX, testY, verbose=2)

    print("Final Performance on test data: ", out)

def main():
    valX = None
    valY = None
    # load dataset
    trainX, trainY, testX, testY = load_dataset()

    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    if use_split_train:
        trainX, valX, trainY, valY = split_train_data(trainX, trainY)

    trainFeatureExtractionLayers(trainX, trainY, valX, valY, testX, testY)


if __name__ == "__main__":
    logger.info(f'Start GLNAS-cifar10.py')
    logger.info(msg=f"saved model path: {CONST_SAVED_MODEL_DIR}")
    logger.info(msg=f"learning_rate: {learning_rate}, use_split_train: {use_split_train}")
    logger.info(msg=f"num_of_layer_feature_extraction: {num_of_layer_feature_extraction}, num_of_layer_classification: {num_of_layer_classification}, useDataAug: {useDataAug}")
    logger.info(msg=f"early_stop: {early_stop}, skipPreResnet: {skipPreResnet}, score_gamma: {score_gamma}, stddev_score: {stddev_score}")
    logger.info(msg=f"useKernelReg: {useKernelReg}, use_separated_loss: {use_separated_loss}, OPTIMIZER: {CONST_OPTIMIZER}")
    logger.info(msg=f"useLayerRestriction: {useLayerRestriction}, use_weight_decay: {use_weight_decay}, repeat: {repeat}, useImgGenerator: {useImgGenerator}")
    logger.info(msg=f"layer_b4_flat: {layer_b4_flat}, eval_performance: {eval_performance}, topk: {top_k}, use_only_add_layer: {use_only_add_layer}")
    logger.info(msg=f"merge_or_add: {merge_or_add}. relu_position: {relu_position}, freeze_pre_layer: {freeze_pre_layer}")
    logger.info(msg=f"use_preprocessing_layer: {use_preprocessing_layer}")
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f'Total time = {(end_time - start_time)}')
    print(f"logger file: {logname}")
    print(f"saved model directory: {CONST_SAVED_MODEL_OUTDIR}")
