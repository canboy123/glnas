import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from tensorflow import keras
from keras import activations
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
import os
import numpy as np
from operations import *
from model import createStartingGLNASModel, addLayerToGLNASModel, createModelManually, BestModelTrainer, createResNetFromGLNASManually
from dataset import datasetBuilder

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

########################################
#            INITIALIZATION            #
########################################
useDataAug = True
# useDataAug = False

useImgGenerator = True
useImgGenerator = False

use_weight_decay = True
use_weight_decay = False

use_new_weight = True
use_new_weight = False

batch_size = 128
epochs = 200
useBlock = True
useBlock = False

repeat = 1

num_classes = 10
input_shape = (32, 32, 3)
flatten_shape = 32 * 32 * 3
inputs = tf.keras.Input(shape=input_shape)

CONST_SAVED_MODEL_PARENT_DIR = "cifar10_tf_e20_202303082200_3"
CONST_LOAD_MODEL_DIR = "saved_model_dir_final"
# CONST_LOAD_MODEL_DIR = "saved_model_dir_best_final"
my_saved_model = f"saved_models/{CONST_SAVED_MODEL_PARENT_DIR}/"+CONST_LOAD_MODEL_DIR

CONST_SAVED_MODEL_OUTDIR = f"saved_models/{CONST_SAVED_MODEL_PARENT_DIR}"
CONST_SAVED_MODEL_DIR = CONST_SAVED_MODEL_OUTDIR+"/"+"saved_model_dir"

useKerasFitTraining = True
useKerasFitTraining = False

if useKerasFitTraining is True and my_saved_model.find("tf") > -1:
    isTf = True
else:
    isTf = False

print(f"isTf: {isTf}, my_saved_model: {my_saved_model}")
print(f"useKerasFitTraining: {useKerasFitTraining}")

weight_decay = 0.0
if use_weight_decay:
    weight_decay = 1e-3

CONST_LR = 0.01

CONST_OPTIMIZER = "sgd"
# CONST_OPTIMIZER = "adam"

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

########################################
#          DATASET FUNCTION            #
########################################
# load train and test dataset
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

########################################
#          SAVE/LOAD FUNCTION          #
########################################
def save_model(model, layer_index):
    print(f"save best final model to {CONST_SAVED_MODEL_DIR}_{layer_index}")
    # Save the trained model
    Path(f"{CONST_SAVED_MODEL_DIR}_{layer_index}").mkdir(parents=True, exist_ok=True)
    try:
        model.save(f"{CONST_SAVED_MODEL_DIR}_{layer_index}", save_format="tf")
    except Exception:
        model.save(f"{CONST_SAVED_MODEL_DIR}_{layer_index}/saved_model.h5")

def load_saved_model(path):
    print("LOAD MODEL PATH", path)
    if path != "" and path is not None:
        model = load_model(f"{path}")

    return model

def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
        for k, initializer in layer.__dict__.items():
            if "initializer" not in k:
                continue
            # find the corresponding variable
            var = getattr(layer, k.replace("_initializer", ""))
            var.assign(initializer(var.shape, var.dtype))

    return model

def lr_schedule(epoch, learning_rate):
    if epoch < 10:
        return learning_rate
    else:
        return learning_rate * tf.math.exp(-0.1)

def getOptimizer():
    if CONST_OPTIMIZER == "adam":
        optimizer = tf.keras.optimizers.Adam(decay=weight_decay)
    else:
        new_learning_rate = CONST_LR

        decay_steps = int(epochs * 50000 / batch_size)
        use_learning_rate = tf.keras.experimental.CosineDecay(new_learning_rate, decay_steps=decay_steps)

        # step_in_1_epoch = 50000 // batch_size
        # step_in_1_epoch = int(step_in_1_epoch)
        # use_learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        #                 [step_in_1_epoch*2, step_in_1_epoch*85],   # In tensorflow, this is based on the step
        #                 [new_learning_rate, new_learning_rate / 10., new_learning_rate / 100.]
        #             )

        optimizer = tf.keras.optimizers.SGD(use_learning_rate, momentum=0.9, decay=weight_decay)

    return optimizer


########################################
#       TRAINING MODEL FUNCTION        #
########################################
@tf.function
def test_step(images, labels, model):
    logits = model(images, training=False)
    multi_softmax = tf.nn.softmax(logits)
    multi_total_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits), 1)

    equalvalent_check = tf.cast(tf.equal(tf.argmax(labels, 2), tf.argmax(logits, 2)), 'float32')
    multi_accuracy = tf.reduce_mean(equalvalent_check, 1)

    return multi_total_loss, multi_accuracy, multi_softmax, equalvalent_check

def start_testing(train_dataset, model, num_of_models):
    total_loss = []
    total_acc = []
    total_logit = []
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # Duplicate the y_batch_train based on the number of models that we have
        y_batch_train = [y_batch_train for i in range(num_of_models)]
        y_batch_train = tf.cast(tf.convert_to_tensor(y_batch_train), tf.int64)
        loss, acc, softmax, equalvalent = test_step(x_batch_train, y_batch_train, model)

        tmp_best_logit = np.max(softmax, 2)

        best_logit = []
        for i in range(len(tmp_best_logit)):
            tmp = np.dot(tmp_best_logit[i], np.transpose(equalvalent[i]))
            best_logit.append(tmp/len(x_batch_train))

        total_loss.append(loss)
        total_acc.append(acc)
        total_logit.append(best_logit)
    total_loss = np.mean(total_loss, 0)
    total_acc = np.mean(total_acc, 0)
    total_logit = np.mean(total_logit, 0)

    return total_loss, total_acc, total_logit

def keras_fit_training(model, x, y=None, testX=None, testY=None, steps_per_epoch=None):
    if type(x) == np.ndarray or y is None:
        model.fit(x, epochs=epochs, batch_size=batch_size, verbose=2, steps_per_epoch=steps_per_epoch,
                  callbacks=[LearningRateScheduler(lr_schedule)])
    else:
        if testX is None and testY is None:
            model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=2,
                      callbacks=[LearningRateScheduler(lr_schedule)])
        else:
            model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=2,
                      callbacks=[LearningRateScheduler(lr_schedule)])

def main():
    # Load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
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
            trainX = datagen.flow(trainX, trainY, batch_size=batch_size)
            trainY = None

            steps_per_epoch = trainX.shape[0] // batch_size
        else:
            builder = datasetBuilder(buffer_size=50000)
            trainX = builder.build_dataset(
                trainY, trainX, batch_size, training=True, repeat=repeat)
            trainY = None
            testX = builder.build_dataset(
                testY, testX, batch_size, training=False)

    GLNASModel = load_saved_model(my_saved_model)

    for layer in GLNASModel.layers:
        layer.trainable = True

    # GLNASModel = createResNetFromGLNASManually((32, 32, 3), 10)

    # Compile and train
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    optimizer = getOptimizer()

    GLNASModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("============== GLNASModel model ===============")
    GLNASModel.summary()
    if useDataAug:
        loss, acc = GLNASModel.evaluate(testX, verbose=2)
    else:
        loss, acc = GLNASModel.evaluate(testX, testY, verbose=2)
    print(f"best accuracy before retraining: {acc}")

    if use_new_weight:
        # Reinitialize the model's weight from the loaded model
        # GLNASModel2 = tf.keras.models.clone_model(GLNASModel)
        if isTf is True:
            last_layer_output = GLNASModel.layers[-1].output
            last_layer_output = tf.keras.activations.softmax(last_layer_output)
            GLNASModel2 = tf.keras.models.Model(inputs=GLNASModel.input, outputs=last_layer_output)
        else:
            GLNASModel2 = tf.keras.models.Model(inputs=GLNASModel.input, outputs=GLNASModel.layers[-1].output)

        GLNASModel2 = reset_weights(GLNASModel2)
        print("============== GLNASModel model2 ===============")
        GLNASModel2.summary()
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        GLNASModel2.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if useKerasFitTraining is True:
            # Keras fit training
            keras_fit_training(GLNASModel2, x=trainX, y=trainY, testX=testX, testY=testY, steps_per_epoch=steps_per_epoch)
        else:
            trainer = BestModelTrainer(GLNASModel2, useImgGenerator, steps_per_epoch)
            trainer.start_training(trainX, epochs, optimizer, batch_size)

        if useDataAug:
            GLNASModel2.evaluate(testX, verbose=2)
        else:
            GLNASModel2.evaluate(testX, testY, verbose=2)
    else:
        if useKerasFitTraining and isTf is True:
            last_layer_output = GLNASModel.layers[-1].output
            last_layer_output = tf.keras.activations.softmax(last_layer_output)
            GLNASModel = tf.keras.models.Model(inputs=GLNASModel.input, outputs=last_layer_output)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
            GLNASModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            print("============== GLNASModel model ===============")
            GLNASModel.summary()

            # Keras fit training
            keras_fit_training(GLNASModel, x=trainX, y=trainY, testX=testX, testY=testY, steps_per_epoch=steps_per_epoch)
        else:
            trainer = BestModelTrainer(GLNASModel, useImgGenerator, steps_per_epoch)
            trainer.start_training(trainX, epochs, optimizer, batch_size)

        if useDataAug:
            GLNASModel.evaluate(testX, verbose=2)
        else:
            GLNASModel.evaluate(testX, testY, verbose=2)

    # if bool_save_model:
    layer_index = "best_final"
    save_model(GLNASModel, layer_index)

if __name__ == "__main__":
    main()