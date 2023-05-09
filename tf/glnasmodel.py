import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, \
AveragePooling2D, Add, ReLU, PReLU, LeakyReLU, DepthwiseConv2D, ELU, GlobalAveragePooling2D, Concatenate

import sys
import os
import time
import numpy as np
from tqdm import tqdm
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from operations import *
from model import createStartingGLNASModel, addLayerToGLNASModel, addClassificationLayerToGLNASModel, Trainer, \
    oneOutputLayerAfterFlatten, createStemForGLNASModel

weight_decay = 3e-4
class Glnas(object):
    """Trains a ResNetCifar10 model."""

    def __init__(self, input_shape, num_class, batch_size, num_of_layer, feature_search_space, classification_search_space,
                 decay_steps, use_kernel_reg=False, layer_b4_flat="average", top_k=1, merge_or_add="add", use_only_add_layer=True,
                 ending_avg_pool_size=2, config=None):
        """Constructor.
        """
        self._input_shape = input_shape
        self._num_class = num_class
        self._batch_size = batch_size
        self._num_of_layer = num_of_layer
        self._feature_search_space = feature_search_space
        self._classification_search_space = classification_search_space
        self._ending_avg_pool_size = ending_avg_pool_size
        self._models = None
        self._best_model = None
        self._use_kernel_reg = use_kernel_reg
        self._kernel_initializer = tf.keras.initializers.he_normal()
        if use_kernel_reg:
            self._kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
        else:
            self._kernel_regularizer = None
        self._config = config
        self._total_models = 1
        self._decay_steps = decay_steps

        self._skip_from_duplicated_layers = {
            "input": [],
            "conv2d": [],
            "convbatch": [],
            "depth2d": [],
            "sepconv": [],
            "dilconv": [],
            "srn": [],
            "elu": ["elu", "relu", "prelu", "leakyrelu"],
            "relu": ["elu", "relu", "prelu", "leakyrelu"],
            "prelu": ["elu", "relu", "prelu", "leakyrelu"],
            "leakyrelu": ["elu", "relu", "prelu", "leakyrelu"],
            "maxpooling2d": ["maxpooling2d", "averagegpool2d"],
            "averagegpool2d": ["maxpooling2d", "averagegpool2d"],
            "convbatch": ["batchnormalization"],
            "batchnormalization": ["batchnormalization"],
            "convbatchavg": ["maxpooling2d", "averagegpool2d"],
            "convbatchmax": ["maxpooling2d", "averagegpool2d"],
            "dropout": ["dropout", "maxpooling2d", "averagegpool2d"],
        }

        self._layer_before_flatten = layer_b4_flat
        self._top_k = top_k
        self._merge_or_add = merge_or_add
        self._use_only_add_layer = use_only_add_layer   # This will work when top k is bigger than 1

        self.loss_object = tf.keras.losses.CategoricalCrossentropy()

        lr = 1e-1

        learning_rate_fn = tf.keras.experimental.CosineDecay(lr, decay_steps=self._decay_steps)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
        self.weight_decay = 5e-4

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    def set_ending_avg_pool_size(self, pool_size):
        self._ending_avg_pool_size = pool_size

    def createFirstNetwork(self, isTf=False, setFilter=32, dataset=None):
        output_activation = "softmax"

        # In tensorflow, we use 'softmax_cross_entropy_with_logit()', so we do not need to use 'softmax' at the output layer
        if isTf:
            output_activation = None

        inputs = Input(shape=self._input_shape)

        if dataset == "imagenet":
            stem = StartingBackboneImgNet(setFilter, self._kernel_initializer, self._kernel_regularizer, "sb_0",
                                          relu_position=self._config["relu_position"])(inputs)
        else:
            stem = StartingBackbone(setFilter, 3, 1, "same", None, self._kernel_initializer, self._kernel_regularizer, "sb_0",
                                    relu_position=self._config["relu_position"])(inputs)

        if self._layer_before_flatten == "average":
            x = AveragePooling2D(pool_size=(self._ending_avg_pool_size, self._ending_avg_pool_size))(stem)
        else:
            x = GlobalAveragePooling2D()(stem)
        x = Flatten()(x)
        x = Dense(self._num_class, activation=output_activation, name=f"classifier_0_0")(x)

        self._models = tf.keras.models.Model(inputs=inputs, outputs=x)
        self._best_model = self._models

    """
    Create a starting model of GLNAS
    """
    def createMiddleNetwork(self, layer_index, isTf=False, setFilter=32, down_sample=False, dataset=None):
        output_activation = "softmax"

        # In tensorflow, we use 'softmax_cross_entropy_with_logit()', so we do not need to use 'softmax' at the output layer
        if isTf:
            output_activation = None

        if self._best_model is not None:
            # Stop the update of the weight for the previous layer
            for layer in self._best_model.layers:
                layer.trainable = False

            inputs = self._best_model.input
            fixed_layer = self._best_model.layers[-4].output
        else:
            inputs = Input(shape=self._input_shape)

            # conv = Conv2D(setFilter, kernel_size=3, strides=1, padding='same', use_bias=False, name="conv2d_0")
            # bn = BatchNormalization(name="batchnormalization_0")
            # relu = ReLU(name="relu_0")
            # fixed_layer = bn(conv(inputs))

            if dataset == "imagenet":
                fixed_layer = StartingBackboneImgNet(setFilter, self._kernel_initializer, self._kernel_regularizer, "sb_0",
                                              relu_position=self._config["relu_position"])(inputs)
            else:
                fixed_layer = StartingBackbone(setFilter, 3, 1, "same", None, self._kernel_initializer,
                                               self._kernel_regularizer, "sb_0", relu_position=self._config["relu_position"])(inputs)

        _ops, _classifier_index = self._add_search_space_op_to_layer(layer_index, setFilter, fixed_layer, output_activation,
                                                                     down_sample, relu_position=self._config["relu_position"])
        self._total_models = _classifier_index

        self._models = tf.keras.models.Model(inputs=inputs, outputs=_ops)
        del inputs, _ops

    def _add_search_space_op_to_layer(self, layer_index, setFilter, fixed_layer, output_activation,
                                      down_sample=False, relu_position="last"):
        pre_layer_name = fixed_layer.name

        # Only get the name that we have defined it, which remove the remaining name at behind
        if pre_layer_name.find("/") >= 0:
            pre_layer_name = pre_layer_name[0:pre_layer_name.find("/")]
        split_pre_layer_name = pre_layer_name.split("_")
        pre_layer_op_name = split_pre_layer_name[0]
        pre_layer_op_num = split_pre_layer_name[-1]

        # If we use block search space, then we do not need to check pre-pre-layer anymore
        # prepre_layer_output = self._best_model.layers[-5].output
        # prepre_layer_name = prepre_layer_output.name
        # split_prepre_layer_name = prepre_layer_name.split("_")
        # split_prepre_layer_name = split_prepre_layer_name[0]

        skip_layers = {}
        if self._config.get("use_layer_restriction") is not None and self._config["use_layer_restriction"] is True \
                and self._skip_from_duplicated_layers.get(pre_layer_op_name) is not None:
            # skip_layers = self._skip_from_duplicated_layers[pre_layer_op_name] + self._skip_from_duplicated_layers[
            #     split_prepre_layer_name]
            skip_layers = self._skip_from_duplicated_layers[pre_layer_op_name]

        _ops = []
        classifier_index = 0
        op_index = 0  # Just for easy to track the layer
        pre_layer_setting = None
        for item in self._feature_search_space:
            for op_name, settings in item.items():
                # Get all settings used on the previous layer for the skip connection later
                if int(pre_layer_op_num) == int(op_index):
                    pre_layer_setting = settings

                # If the previous layer is maxpooling, averagepooling, batchnormalization, convbatchmax or dropout,
                # then we skip some specified layers with the similar categories
                if op_name in skip_layers:
                    continue
                # isSkip = False
                # for skip_layer in skip_layers:
                #     if skip_layer == op_name:
                #         isSkip = True
                #         break
                #
                # if isSkip:
                #     continue

                filters = setFilter
                kernel = settings.get("kernel")
                if down_sample is False:
                    stride = settings.get("stride")
                else:
                    stride = 2
                activation = settings.get("activation")
                padding = settings.get("padding")
                pool = settings.get("pool")
                rate = settings.get("rate")
                unit = settings.get("unit")
                kernel_initializer = self._kernel_initializer
                kernel_regularizer = self._kernel_regularizer
                name = f"{op_name}_{layer_index}_{op_index}"

                op = OPS[op_name](filters, kernel, stride, activation, padding, pool, rate, unit,
                                  kernel_initializer, kernel_regularizer, name, self._config["relu_position"])

                x = op(fixed_layer)
                if self._layer_before_flatten == "average":
                    x = AveragePooling2D(pool_size=(self._ending_avg_pool_size, self._ending_avg_pool_size))(x)
                else:
                    x = GlobalAveragePooling2D()(x)
                x = Flatten()(x)
                x = Dense(self._num_class, activation=output_activation, name=f"classifier_{layer_index}_{classifier_index}")(x)

                _ops.append(x)
                classifier_index += 1
            op_index += 1

        ################################
        ###     SKIP CONNECTION      ###
        ################################
        if self._config.get("use_resnet_mode") is not None and self._config["use_resnet_mode"] is True:
            # Resnet style only for conv2d or batchnormalization
            tmp_model = tf.keras.models.Model(inputs=self._best_model.input, outputs=fixed_layer)

            tmp_keep_layers = []
            tmp_keep_layer_names = []
            for layer in tmp_model.layers:
                tmp_name = layer.name

                if tmp_name.find("input") != -1:
                    continue

                split_layer_name = tmp_name.split("_")
                split_layer_name = split_layer_name[0]

                # Skip all previous layers if the previous layer is resnet layer
                if split_layer_name == "srn" and self._config.get("skip_preresnet") is not None and self._config["skip_preresnet"] is True:
                    tmp_keep_layers = []
                    tmp_keep_layer_names = []

                # If the top k is bigger than 1, then we just connect with the add layer
                if self._top_k > 1 and self._use_only_add_layer is True:
                    if split_layer_name == "add" or split_layer_name == "concat":
                        tmp_keep_layers.append(layer.output)
                        tmp_keep_layer_names.append(tmp_name)
                else:
                    tmp_keep_layers.append(layer.output)
                    tmp_keep_layer_names.append(tmp_name)

            # Last layer is the same as the previous layer, we do not need to do the skip network for the previous layer
            if len(tmp_keep_layers) > 0:
                tmp_keep_layers.pop()
                tmp_keep_layer_names.pop()

            print("tmp_keep_layers", len(tmp_keep_layer_names), tmp_keep_layer_names)

            del tmp_model

            for i in range(len(tmp_keep_layers)):
                layer_output = tmp_keep_layers[i]
                split_layer_name = tmp_keep_layer_names[i]
                filters = setFilter
                kernel = (3, 3)
                if down_sample is False:
                    stride = 1
                else:
                    stride = 2
                padding = "same"
                kernel_initializer = self._kernel_initializer
                kernel_regularizer = self._kernel_regularizer

                name = f"srn_{split_layer_name}_{layer_index}_{op_index}"

                input_shape1 = fixed_layer.shape[1]
                input_shape2 = layer_output.shape[1]

                # Reduce the size of the second input if they are different
                num_of_pool = 0
                if input_shape1 != input_shape2:
                    num_of_pool = int(math.log(input_shape2 / input_shape1) / math.log(2))

                # print(layer_output, name)
                op = SimpleResNet(filters, kernel, stride, padding, name, layer_output, kernel_initializer, kernel_regularizer,
                                  pre_layer_setting, down_sample, num_of_pool, relu_position=relu_position)

                x = op(fixed_layer, layer_output)
                if self._layer_before_flatten == "average":
                    x = AveragePooling2D(pool_size=(self._ending_avg_pool_size, self._ending_avg_pool_size))(x)
                else:
                    x = GlobalAveragePooling2D()(x)
                x = Flatten()(x)
                x = Dense(self._num_class, activation=output_activation, name=f"classifier_{layer_index}_{classifier_index}")(x)

                _ops.append(x)
                classifier_index += 1
                op_index += 1

        return _ops, classifier_index

    def selectBestModel(self, layer_index, best_index):
        print("Get Best Model")
        if len(best_index) == 1:
            b_index = best_index[0]
            self._best_model = tf.keras.models.Model(inputs=self._models.input, outputs=self._models.get_layer(f'classifier_{layer_index}_{b_index}').output)
        else:
            models = []
            for i in range(len(best_index)):
                b_index = best_index[i]
                tmpModel = tf.keras.models.Model(inputs=self._models.input, outputs=self._models.get_layer(f'classifier_{layer_index}_{b_index}').output)
                tmpModel = tmpModel.layers[-4].output
                models.append(tmpModel)

            if self._merge_or_add == "add":
                merge = Add(name=f"add_{layer_index}")
            else:
                merge = Concatenate(name=f"concat_{layer_index}")
            x = merge(models)
            if self._layer_before_flatten == "average":
                x = AveragePooling2D(pool_size=(self._ending_avg_pool_size, self._ending_avg_pool_size))(x)
            else:
                x = GlobalAveragePooling2D()(x)
            x = Flatten()(x)
            output = Dense(self._num_class, name=f"classifier_{layer_index}_0")(x)

            self._best_model = tf.keras.models.Model(inputs=self._models.input, outputs=output)

    def getSubModel(self, layer_index, output_index):
        sub_model = tf.keras.models.Model(inputs=self._models.input, outputs=self._models.get_layer(f'classifier_{layer_index}_{output_index}').output)
        return sub_model

    def getOptimizer(self, epochs, cur_num_layer, learning_rate, use_optimizer="sgd"):
        if use_optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(decay=self.weight_decay)
        else:
            if cur_num_layer == 0:
                cur_num_layer = 1
            new_learning_rate = learning_rate / cur_num_layer

            step_in_1_epoch = 50000 // self._batch_size

            step_in_1_epoch = int(step_in_1_epoch)

            use_learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                [step_in_1_epoch * 2, step_in_1_epoch * 85],  # In tensorflow, this is based on the step
                [new_learning_rate, new_learning_rate / 10., new_learning_rate / 100.]
            )
            decay_steps = int(epochs * 50000 / self._batch_size)
            use_learning_rate = tf.keras.experimental.CosineDecay(new_learning_rate, decay_steps=decay_steps)
            optimizer = tf.keras.optimizers.SGD(use_learning_rate, momentum=0.9)
            # optimizer = tf.keras.optimizers.SGD(new_learning_rate, momentum=0.9)

        return optimizer

    def start_training(self, train_dataset, test_dataset, epochs, layer_index, optimizer):
        models_loss = []
        models_acc = []
        models_logits = []
        models = []
        optimizers = []
        for i in range(self._total_models):
            sub_model = self.getSubModel(layer_index, i)
            models.append(sub_model)
            optimizer = self.getOptimizer(epochs, layer_index, 0.01)
            optimizers.append(optimizer)

        trainer = Trainer(models, self._total_models, i, optimizers, layer_index)
        trainer.start_training(train_dataset, epochs)
        loss, acc, logits = trainer.start_testing(test_dataset)

        del trainer
        del models
        del optimizers
        models_loss.append(loss)
        models_acc.append(acc)
        models_logits.append(logits)

        return acc, loss, logits

    def getBestIndex(self, acc, loss, logit):
        new_acc = np.array(acc)
        new_logit = np.array(logit)

        top_index_logit = np.argpartition(new_logit, -self._top_k)[-self._top_k:]
        top_index_acc = np.argpartition(new_acc, -self._top_k)[-self._top_k:]

        best_index_acc = np.argmax(new_acc)
        best_index_logit = np.nanargmax(new_logit)

        return top_index_logit, top_index_acc, best_index_acc, best_index_logit


    def eval_best_model(self, test_dataset):
        trainer = Trainer(self._best_model, 1, 0, None)
        loss, acc, logits = trainer.start_testing(test_dataset)
        print(f"Best Model Evaluation")
        print(f"Acc: {acc}")
        print(f"Loss: {loss}")

    def get_best_model(self):
        return self._best_model

    def set_best_model(self, best_model):
        self._best_model = best_model