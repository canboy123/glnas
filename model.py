import tensorflow as tf
import numpy as np
from operations import *
import time
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, \
AveragePooling2D, Add, ReLU, PReLU, LeakyReLU, DepthwiseConv2D, ELU, GlobalAveragePooling2D, Concatenate

CONST_REGULARIZER = tf.keras.regularizers.l2(1e-4)
CONST_AVG_POOLING_POOL_SIZE = 2


class BestModelTrainer(object):
    """Trains a ResNetCifar10 model."""

    def __init__(self, model, useImgGenerator=False, steps_per_epoch=None):
        """Constructor.

        Args:
          model: an instance of ResNetCifar10Model instance.
        """
        self._model = model
        self._use_img_generator = useImgGenerator
        self._steps_per_epoch = steps_per_epoch
        self.weight_decay = 5e-4

        # self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        # self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')


    def start_training(self, dataset, epochs, optimizer, batch_size):
        @tf.function()
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = self._model(images, training=True)
                cross_entropy_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels, logits=logits))
                # regularization_losses = self._model.losses
                regularization_losses = tf.add_n([tf.nn.l2_loss(v) for v in self._model.trainable_variables])
                # total_loss = tf.add_n(regularization_losses + [cross_entropy_loss])
                total_loss = cross_entropy_loss + regularization_losses*self.weight_decay

                # losses = {}
                # losses['reg'] = tf.reduce_sum(self._model.losses)
                # losses['ce'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
                # total_loss = tf.add_n([l for l in losses.values()])

            accuracy = tf.reduce_mean(tf.cast(tf.equal(
                tf.argmax(labels, 1), tf.argmax(logits, 1)), 'float32'))
            gradients = tape.gradient(total_loss, self._model.trainable_variables)
            gradients = [(tf.clip_by_norm(grad, 5.0)) for grad in gradients]

            optimizer.apply_gradients(
                zip(gradients, self._model.trainable_variables))
            step = optimizer.iterations
            lr = optimizer.learning_rate(step)

            return total_loss, accuracy, step - 1, lr

        print('Training from scratch...')

        for epoch in range(epochs):
            avg_acc = []
            avg_loss = []
            for step, (images, labels) in enumerate(dataset):
                total_loss, accuracy, step, lr = train_step(images, labels)
                avg_acc.append(accuracy.numpy())
                avg_loss.append(total_loss.numpy())
                if self._use_img_generator is True and step >= self._steps_per_epoch:
                    break

            print('epoch: %d, loss: %f, accuracy: %f, lr: %f' % (
                epoch, np.average(avg_loss), np.average(avg_acc), lr.numpy()))

class Trainer(object):
    """Trains a ResNetCifar10 model."""

    def __init__(self, models, num_of_models, cur_model_index, optimizers, layer_index):
        """Constructor.

        Args:
          model: an instance of ResNetCifar10Model instance.
        """
        self._models = models
        self._num_of_models = num_of_models
        self._cur_model_index = cur_model_index
        self._optimizers = optimizers
        self._layer_index = layer_index

        self.weight_decay = 5e-4

    def start_training(self, train_dataset, epochs):
        @tf.function()
        def train_step(images, labels):
            loss_list = []
            acc_list = []
            for i in range(len(self._models)):
                model = self._models[i]
                optimizer = self._optimizers[i]
                with tf.GradientTape() as tape:
                    logits = model(images, training=True)

                    # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
                    # regularization_losses = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
                    # total_loss = cross_entropy_loss + regularization_losses*self.weight_decay
                    # total_loss = cross_entropy_loss

                    losses = {}
                    losses['reg'] = tf.reduce_sum(model.losses)
                    losses['ce'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
                    total_loss = tf.add_n([l for l in losses.values()])

                    accuracy = tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)), 'float32')
                    accuracy = tf.reduce_mean(accuracy)

                    gradients = tape.gradient(total_loss, model.trainable_variables)
                    gradients = [(tf.clip_by_norm(grad, 5.0)) for grad in gradients]

                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    step = optimizer.iterations
                    lr = optimizer.learning_rate(step)

                    loss_list.append(total_loss)
                    acc_list.append(accuracy)

            return loss_list, acc_list, lr

        for epoch in range(epochs):
            start_time = time.time()
            total_loss = []
            total_acc = []
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                # Duplicate the y_batch_train based on the number of models that we have
                loss, acc, lr = train_step(x_batch_train, y_batch_train)
                total_loss.append(loss)
                total_acc.append(acc)

            end_time = time.time()
            total_time_used = end_time - start_time
            print(f"\nModel ({self._layer_index}): {self._cur_model_index+1} / {self._num_of_models}")
            print(f"learning_rate: {lr}")
            print("Epoch %d" % (epoch,))
            print(f"Time used: {total_time_used}s")
            print(f"Loss: {np.mean(total_loss, 0)}")
            print(f"Acc: {np.mean(total_acc, 0)}")

    def start_testing(self, dataset):
        @tf.function
        def test_step(images, labels):
            loss_list = []
            acc_list = []
            softmax_list = []
            equalvalent_check_list = []
            for i in range(len(self._models)):
                model = self._models[i]
                logits = model(images, training=False)
                softmax = tf.nn.softmax(logits)
                total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
                equalvalent_check = tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)), 'float32')
                accuracy = tf.reduce_mean(equalvalent_check)

                loss_list.append(total_loss)
                acc_list.append(accuracy)
                softmax_list.append(softmax)
                equalvalent_check_list.append(equalvalent_check)

            return loss_list, acc_list, softmax_list, equalvalent_check_list

        total_loss = []
        total_acc = []
        total_logit = []
        # Iterate over the batches of the dataset.
        for step, (x_batch, y_batch) in enumerate(dataset):
            # Duplicate the y_batch_train based on the number of models that we have
            loss, acc, softmax, equalvalent = test_step(x_batch, y_batch)

            equalvalent = np.array(equalvalent)
            equalvalent.astype("int64")
            for i in range(len(equalvalent)):
                equalvalent[i] = [j if j != 0 else -1 for j in equalvalent[i]]
            equalvalent = np.array(equalvalent)

            tmp_best_logit = np.max(softmax, 2)


            equalvalent.astype("float32")
            best_logit = []
            for i in range(len(tmp_best_logit)):
                tmp = np.dot(tmp_best_logit[i], np.transpose(equalvalent[i]))
                best_logit.append(tmp/len(x_batch))
            # print(f"loss: {loss}")
            # print(f"acc: {acc}")
            # print(f"softmax: {softmax}")
            # print(f"equalvalent: {equalvalent}")
            # print("tmp", tmp)
            # print("softmax", softmax)
            # print("tmp_best_logit", tmp_best_logit)
            # print("equalvalent", np.shape(equalvalent), equalvalent)

            total_loss.append(loss)
            total_acc.append(acc)
            total_logit.append(best_logit)
        total_loss = np.mean(total_loss, 0)
        total_acc = np.mean(total_acc, 0)
        total_logit = np.mean(total_logit, 0)

        return total_loss, total_acc, total_logit


"""
Create a starting model of GLNAS
"""
def createStemForGLNASModel(input_shape, num_class, isTf=False, setFilter=32):
    output_activation = "softmax"

    # In tensorflow, we use 'softmax_cross_entropy_with_logit()', so we do not need to use 'softmax' at the output layer
    if isTf:
        output_activation = None

    inputs = tf.keras.layers.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(setFilter, kernel_size=3, strides=1, padding='same', use_bias=False, name="conv2d_0")
    bn1 = tf.keras.layers.BatchNormalization(name="batchnormalization_0")
    relu1 = tf.keras.layers.ReLU(name="relu_0")
    stem = relu1(bn1(conv1(inputs)))
    x = tf.keras.layers.Flatten()(stem)
    x = tf.keras.layers.Dense(num_class, activation=output_activation, name=f"classifier_0_0")(x)

    GLNASModel = tf.keras.models.Model(inputs=inputs, outputs=x)

    return GLNASModel

"""
Create a starting model of GLNAS
"""
def createStartingGLNASModel(useSearchSpace, input_shape, num_class, useKernelReg=False, isTf=False, setFilter=32, stem_network=None):
    output_activation = "softmax"

    # In tensorflow, we use 'softmax_cross_entropy_with_logit()', so we do not need to use 'softmax' at the output layer
    if isTf:
        output_activation = None

    if stem_network is not None:
        # Stop the update of the weight for the previous layer
        for layer in stem_network.layers:
            layer.trainable = False

        inputs = stem_network.input

        stem = stem_network.layers[-3].output
    else:
        inputs = tf.keras.layers.Input(shape=input_shape)

        conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)
        bn1 = tf.keras.layers.BatchNormalization()
        stem = tf.keras.activations.relu(bn1(conv1(inputs)))

    _ops = []
    for i in range(len(useSearchSpace)):
        for op_name, settings in useSearchSpace[i].items():
            filters = setFilter
            kernel = settings.get("kernel")
            stride = settings.get("stride")
            activation = settings.get("activation")
            padding = settings.get("padding")
            pool = settings.get("pool")
            rate = settings.get("rate")
            unit = settings.get("unit")
            if useKernelReg:
                kernel_regularizer = tf.keras.regularizers.l2(1e-4)
            else:
                kernel_regularizer = None
            name = f"{op_name}_1_{i}"

            op = OPS[op_name](filters, kernel, stride, activation, padding, pool, rate, unit, kernel_regularizer, name)

            x = op(stem)
            x = tf.keras.layers.AveragePooling2D(pool_size=CONST_AVG_POOLING_POOL_SIZE)(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(num_class, activation=output_activation, name=f"classifier_1_{i}")(x)

            _ops.append(x)

    GLNASModel = tf.keras.models.Model(inputs=inputs, outputs=_ops)
    del inputs, _ops

    return GLNASModel


def addLayerToGLNASModel(useSearchSpace, GLNASModel, num_class, layer_index, useResnetMode=False,
                         useKernelReg=False, isTf=False, skipPreResnet=True, useLayerRestriction=True,
                         setFilter=32):
    output_activation = "softmax"

    # In tensorflow, we use 'softmax_cross_entropy_with_logit()', so we do not need to use 'softmax' at the output layer
    if isTf:
        output_activation = None

    inputs = GLNASModel.input

    # Stop the update of the weight for the previous layer
    for layer in GLNASModel.layers:
        layer.trainable = False

    pre_layer_output = GLNASModel.layers[-4].output
    pre_layer_name = pre_layer_output.name
    # Only get the name that we have defined it, which remove the remaining name at behind
    if pre_layer_name.find("/") >= 0:
        pre_layer_name = pre_layer_name[0:pre_layer_name.find("/")]
    split_pre_layer_name = pre_layer_name.split("_")
    pre_layer_op_name = split_pre_layer_name[0]
    pre_layer_op_num = split_pre_layer_name[-1]

    prepre_layer_output = GLNASModel.layers[-5].output
    prepre_layer_name = prepre_layer_output.name
    split_prepre_layer_name = prepre_layer_name.split("_")
    split_prepre_layer_name = split_prepre_layer_name[0]

    skip_from_duplicated_layers = {
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
        "convbatchmax": ["maxpooling2d", "averagegpool2d"],
        "dropout": ["dropout", "maxpooling2d", "averagegpool2d"],
    }


    skip_layers = {}
    if useLayerRestriction is True and skip_from_duplicated_layers.get(pre_layer_op_name) is not None:
        skip_layers = skip_from_duplicated_layers[pre_layer_op_name] + skip_from_duplicated_layers[split_prepre_layer_name]

    _ops = []
    pre_layer_setting = None
    # for i in range(len(useSearchSpace)):
    classifier_index = 0
    op_index = 0 # Just for easy to track the layer
    for item in useSearchSpace:
        for op_name, settings in item.items():
            # Get all settings used on the previous layer for the skip connection later
            if int(pre_layer_op_num) == int(op_index):
                pre_layer_setting = settings

            isSkip = False
            # If the previous layer is maxpooling, averagepooling, batchnormalization, convbatchmax or dropout,
            # then we skip some specified layers with the similar categories
            for skip_layer in skip_layers:
                if skip_layer == op_name:
                    isSkip = True
                    break

            if isSkip:
                continue

            filters = setFilter
            kernel = settings.get("kernel")
            stride = settings.get("stride")
            activation = settings.get("activation")
            padding = settings.get("padding")
            pool = settings.get("pool")
            rate = settings.get("rate")
            unit = settings.get("unit")
            if useKernelReg:
                kernel_regularizer = CONST_REGULARIZER
            else:
                kernel_regularizer = None
            name = f"{op_name}_{layer_index}_{op_index}"

            op = OPS[op_name](filters, kernel, stride, activation, padding, pool, rate, unit, kernel_regularizer, name)

            x = op(pre_layer_output)
            x = tf.keras.layers.AveragePooling2D(pool_size=CONST_AVG_POOLING_POOL_SIZE)(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(num_class, activation=output_activation, name=f"classifier_{layer_index}_{classifier_index}")(x)

            _ops.append(x)
            classifier_index += 1
        op_index += 1

    ################################
    ###     SKIP CONNECTION      ###
    ################################
    if useResnetMode:
        # Resnet style only for conv2d or batchnormalization
        tmp_model = tf.keras.models.Model(inputs=inputs, outputs=pre_layer_output)

        tmp_keep_layers = []
        tmp_keep_layer_names = []
        for layer in tmp_model.layers:
            tmp_name = layer.name

            if tmp_name.find("input") != -1:
                continue

            split_layer_name = tmp_name.split("_")
            split_layer_name = split_layer_name[0]

            # Skip all previous layers if the previous layer is resnet layer
            if split_layer_name == "srn" and skipPreResnet is True:
                tmp_keep_layers = []
                tmp_keep_layer_names = []

            tmp_keep_layers.append(layer.output)
            tmp_keep_layer_names.append(tmp_name)

        # Last layer is the same as the previous layer, we do not need to do the skip network for the previous layer
        if len(tmp_keep_layers) > 0:
            tmp_keep_layers.pop()
            tmp_keep_layer_names.pop()

        print("tmp_keep_layers", len(tmp_keep_layer_names), tmp_keep_layer_names)

        for i in range(len(tmp_keep_layers)):
            layer_output = tmp_keep_layers[i]
            split_layer_name = tmp_keep_layer_names[i]
            filters = setFilter
            kernel = (3, 3)
            stride = 1
            padding = "same"
            if useKernelReg:
                kernel_regularizer = CONST_REGULARIZER
            else:
                kernel_regularizer = None
            name = f"srn_{split_layer_name}_{layer_index}_{op_index}"

            # print(layer_output, name)
            op = SimpleResNet(filters, kernel, stride, padding, name, layer_output, kernel_regularizer, pre_layer_setting)


            x = op(pre_layer_output, layer_output)
            x = tf.keras.layers.AveragePooling2D(pool_size=CONST_AVG_POOLING_POOL_SIZE)(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(num_class, activation=output_activation, name=f"classifier_{layer_index}_{classifier_index}")(x)

            _ops.append(x)
            classifier_index += 1
            op_index += 1

    new_model = tf.keras.models.Model(inputs=inputs, outputs=_ops)
    # new_model.summary()

    return new_model, classifier_index

def addClassificationLayerToGLNASModel(useSearchSpace, GLNASModel, num_class, layer_index, useKernelReg=False, isTf=False):
    output_activation = "softmax"

    # In tensorflow, we use 'softmax_cross_entropy_with_logit()', so we do not need to use 'softmax' at the output layer
    if isTf:
        output_activation = None

    inputs = GLNASModel.input

    # Stop the update of the weight for the previous layer
    for layer in GLNASModel.layers:
        layer.trainable = False

    pre_layer_output = GLNASModel.layers[-2].output
    pre_layer_name = pre_layer_output.name
    split_pre_layer_name = pre_layer_name.split("_")
    split_pre_layer_name = split_pre_layer_name[0]

    skip_from_duplicated_layers = {
        "flatten": [],
        "dense": [],
        "batchnormalization": ["batchnormalization"],
        "dropout": ["dropout"],
    }

    skip_layers = {}
    if skip_from_duplicated_layers.get(split_pre_layer_name) is not None:
        skip_layers = skip_from_duplicated_layers[split_pre_layer_name]

    _ops = []
    classifier_index = 0
    op_index = 0 # Just for easy to track the layer
    for i in range(len(useSearchSpace)):
        for op_name, settings in useSearchSpace[i].items():
            isSkip = False
            # If the previous layer is batchnormalization or dropout,
            # then we skip some specified layers with the similar categories
            for skip_layer in skip_layers:
                if skip_layer == op_name:
                    isSkip = True
                    break

            if isSkip:
                continue
            filters = settings.get("filters")
            kernel = settings.get("kernel")
            stride = settings.get("stride")
            activation = settings.get("activation")
            padding = settings.get("padding")
            pool = settings.get("pool")
            rate = settings.get("rate")
            unit = settings.get("unit")
            if useKernelReg:
                kernel_regularizer = tf.keras.regularizers.l2(2e-4)
            else:
                kernel_regularizer = None
            name = f"{op_name}_{layer_index}_{op_index}"

            op = OPS[op_name](filters, kernel, stride, activation, padding, pool, rate, unit, kernel_regularizer, name)

            x = op(pre_layer_output)
            x = tf.keras.layers.Dense(num_class, activation=output_activation, name=f"classifier_{layer_index}_{classifier_index}")(x)

            _ops.append(x)
            classifier_index += 1
        op_index += 1

    new_model = tf.keras.models.Model(inputs=inputs, outputs=_ops)

    return new_model, classifier_index

def oneOutputLayerAfterFlatten(GLNASModel, num_class, layer_index, isTf=False):
    output_activation = "softmax"

    # In tensorflow, we use 'softmax_cross_entropy_with_logit()', so we do not need to use 'softmax' at the output layer
    if isTf:
        output_activation = None

    inputs = GLNASModel.input

    # Stop the update of the weight for the previous layer
    for layer in GLNASModel.layers:
        layer.trainable = False
        # layer.trainable = True

    pre_layer_output = GLNASModel.layers[-1].output

    x = tf.keras.layers.Flatten()(pre_layer_output)
    output = tf.keras.layers.Dense(num_class, activation=output_activation, name=f"classifier_{layer_index}")(x)
    new_model = tf.keras.models.Model(inputs=inputs, outputs=output)

    return new_model

def createModelManually(featureSearchSpace, classificationSearchSpace, input_shape, num_class):
    inputs = tf.keras.layers.Input(shape=input_shape)
    layerIndex = [20, 28, 20, 10, 8, 19, 9, 0, 7, 7]
    layerIndex2 = [4, 1, 2]
    _ops = []
    x = inputs
    for i in range(len(layerIndex)):
        index = layerIndex[i]
        for op_name, settings in featureSearchSpace[index].items():
            filters = settings.get("filters")
            kernel = settings.get("kernel")
            stride = settings.get("stride")
            activation = settings.get("activation")
            padding = settings.get("padding")
            pool = settings.get("pool")
            rate = settings.get("rate")
            unit = settings.get("unit")
            name = f"{op_name}_{i+1}_{index}"

            op = OPS[op_name](filters, kernel, stride, activation, padding, pool, rate, unit, name)
            x = op(x)

    x = tf.keras.layers.Flatten()(x)

    for i in range(len(layerIndex2)):
        index = layerIndex2[i]
        for op_name, settings in classificationSearchSpace[index].items():
            filters = settings.get("filters")
            kernel = settings.get("kernel")
            stride = settings.get("stride")
            activation = settings.get("activation")
            padding = settings.get("padding")
            pool = settings.get("pool")
            rate = settings.get("rate")
            unit = settings.get("unit")
            name = f"{op_name}_{i+len(layerIndex)+1}_{index}"

            op = OPS[op_name](filters, kernel, stride, activation, padding, pool, rate, unit, name)
            x = op(x)


    x = tf.keras.layers.Dense(num_class, activation="softmax", name=f"classifier_1")(x)

    GLNASModel = tf.keras.models.Model(inputs=inputs, outputs=x)

    return GLNASModel

def createManually(input_shape, num_class):
    inputs = tf.keras.layers.Input(shape=input_shape)

    kernel_initializer = tf.keras.initializers.he_normal()
    kernel_regularizer = tf.keras.regularizers.l2(1e-4)

    x1 = StartingBackboneImgNet(16, kernel_initializer, kernel_regularizer, "sb_0")(inputs)

    # cifar10_tf_e20_202303151700_0
    # x11 = Conv2dBatchNorm(16, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_1", "first")(x1)
    # x12 = Conv2dBatchNormMax(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_1", "first")(x1)
    # x13 = DilConv(16, 3, 1, "same", kernel_initializer, kernel_regularizer, "dc_1", True, "first")(x1)
    # x14 = SepConv(16, 3, 1, "same", kernel_initializer, kernel_regularizer, "sc_1", True, "first")(x1)
    # x2 = Concatenate()([x11, x12, x13, x14])
    #
    # x21 = Conv2dBatchNormMax(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_2", "first")(x2)
    # x22 = SimpleResNet(16, 3, 1, "same", "srn_21", x2, kernel_initializer, kernel_regularizer, relu_position="first")(x2, x14)
    # x23 = SimpleResNet(16, 3, 1, "same", "srn_22", x2, kernel_initializer, kernel_regularizer, relu_position="first")(x2, x11)
    # x24 = SimpleResNet(16, 3, 1, "same", "srn_23", x2, kernel_initializer, kernel_regularizer, relu_position="first")(x2, x1)
    # x3 = Concatenate()([x21, x22, x23, x24])
    #
    # x31 = Conv2dBatchNorm(16, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_3", "first")(x3)
    # x32 = Conv2dBatchNormAvg(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_3", "first")(x3)
    # x33 = Conv2dBatchNormMax(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_3", "first")(x3)
    # x34 = SimpleResNet(16, 3, 1, "same", "srn_3", x3, kernel_initializer, kernel_regularizer, relu_position="first")(x3, x24)
    # x4 = Concatenate()([x31, x32, x33, x34])
    #
    # x41 = SimpleResNet(16, 3, 1, "same", "srn_4", x4, kernel_initializer, kernel_regularizer, relu_position="first")(x4,x34)
    # x42 = Conv2dBatchNormAvg(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_4", "first")(x4)
    # x43 = Conv2dBatchNorm(16, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_4", "first")(x4)
    # x44 = Conv2dBatchNormMax(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_4", "first")(x4)
    # x5 = Concatenate()([x41, x42, x43, x44])
    #
    # x51 = SimpleResNet(16, 3, 1, "same", "srn_5", x5, kernel_initializer, kernel_regularizer, relu_position="first")(x5,x43)
    # x52 = Conv2dBatchNorm(16, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_5", "first")(x5)
    # x53 = Conv2dBatchNormAvg(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_5", "first")(x5)
    # x54 = Conv2dBatchNormMax(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_5", "first")(x5)
    # x6 = Concatenate()([x51, x52, x53, x54])
    #
    # x61 = SimpleResNet(32, 3, 2, "same", "srn_6", x6, kernel_initializer, kernel_regularizer, down_sample=True, relu_position="first")(x6,x54)
    # x62 = Conv2dBatchNormAvg(32, 3, 2, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_6", "first")(x6)
    # x63 = Conv2dBatchNormMax(32, 3, 2, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_6", "first")(x6)
    # x64 = Conv2dBatchNorm(32, 3, 2, "same", None, kernel_initializer, kernel_regularizer, "cb_6", "first")(x6)
    # x7 = Concatenate()([x61, x62, x63, x64])
    #
    # x71 = SimpleResNet(32, 3, 1, "same", "srn_71", x7, kernel_initializer, kernel_regularizer, relu_position="first")(x7,x62)
    # x72 = Conv2dBatchNormMax(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_7", "first")(x7)
    # x73 = SimpleResNet(32, 3, 1, "same", "srn_72", x7, kernel_initializer, kernel_regularizer, relu_position="first")(x7,x63)
    # x74 = Conv2dBatchNorm(32, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_7", "first")(x7)
    # x8 = Concatenate()([x71, x72, x73, x74])
    #
    # x81 = SimpleResNet(32, 3, 1, "same", "srn_8", x8, kernel_initializer, kernel_regularizer, relu_position="first")(x8,x73)
    # x82 = Conv2dBatchNormAvg(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_8", "first")(x8)
    # x83 = Conv2dBatchNorm(32, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_8", "first")(x8)
    # x84 = Conv2dBatchNormMax(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_8", "first")(x8)
    # x9 = Concatenate()([x81, x82, x83, x84])
    #
    # x91 = SimpleResNet(32, 3, 1, "same", "srn_9", x9, kernel_initializer, kernel_regularizer, relu_position="first")(x9,x84)
    # x92 = Conv2dBatchNormAvg(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_9", "first")(x9)
    # x93 = Conv2dBatchNormMax(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_9", "first")(x9)
    # x94 = Conv2dBatchNorm(32, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_9", "first")(x9)
    # x10 = Concatenate()([x91, x92, x93, x94])
    #
    # x11 = Conv2dBatchNormAvg(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_10", "first")(x10)
    # x12 = SimpleResNet(32, 3, 1, "same", "srn_10", x10, kernel_initializer, kernel_regularizer, relu_position="first")(x10,x91)
    # x13 = Conv2dBatchNormMax(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_10", "first")(x10)
    # x14 = Conv2dBatchNorm(32, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_10", "first")(x10)
    # x1 = Concatenate()([x11, x12, x13, x14])
    #
    # x21 = SimpleResNet(64, 3, 2, "same", "srn_11", x1, kernel_initializer, kernel_regularizer, down_sample=True, relu_position="first")(x1,x12)
    # x22 = Conv2dBatchNorm(64, 3, 2, "same", None, kernel_initializer, kernel_regularizer, "cb_11", "first")(x1)
    # x23 = Conv2dBatchNormMax(64, 3, 2, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_11", "first")(x1)
    # x24 = Conv2dBatchNormAvg(64, 3, 2, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_11", "first")(x1)
    # x2 = Concatenate()([x21, x22, x23, x24])
    #
    # x31 = SimpleResNet(64, 3, 1, "same", "srn_12", x2, kernel_initializer, kernel_regularizer, relu_position="first")(x2,x22)
    # x32 = Conv2dBatchNorm(64, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_12", "first")(x2)
    # x33 = Conv2dBatchNormMax(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_12", "first")(x2)
    # x34 = Conv2dBatchNormAvg(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_12", "first")(x2)
    # x3 = Concatenate()([x31, x32, x33, x34])
    #
    # x41 = SimpleResNet(64, 3, 1, "same", "srn_13", x3, kernel_initializer, kernel_regularizer, relu_position="first")(x3,x32)
    # x42 = Conv2dBatchNorm(64, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_13", "first")(x3)
    # x43 = Conv2dBatchNormMax(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_13", "first")(x3)
    # x44 = Conv2dBatchNormAvg(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_13", "first")(x3)
    # x4 = Concatenate()([x41, x42, x43, x44])
    #
    # x51 = SimpleResNet(64, 3, 1, "same", "srn_14", x4, kernel_initializer, kernel_regularizer, relu_position="first")(x4,x41)
    # x52 = Conv2dBatchNorm(64, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_14", "first")(x4)
    # x53 = Conv2dBatchNormMax(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_14", "first")(x4)
    # x54 = Conv2dBatchNormAvg(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_14", "first")(x4)
    # x5 = Concatenate()([x51, x52, x53, x54])
    #
    # x61 = SimpleResNet(64, 3, 1, "same", "srn_15", x5, kernel_initializer, kernel_regularizer, relu_position="first")(x5,x52)
    # x62 = Conv2dBatchNormAvg(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_15", "first")(x5)
    # x63 = Conv2dBatchNorm(64, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_15", "first")(x5)
    # x64 = Conv2dBatchNormMax(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_15", "first")(x5)
    # x6 = Concatenate()([x61, x62, x63, x64])



    # cifar10_tf_e20_202304011900_0
    x11 = Conv2dBatchNormAvg(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_1", "first")(x1)
    x12 = Conv2dBatchNormMax(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_1", "first")(x1)
    x13 = DilConv(16, 3, 1, "same", kernel_initializer, kernel_regularizer, "dc_1", True, "first")(x1)
    x14 = SepConv(16, 3, 1, "same", kernel_initializer, kernel_regularizer, "sc_1", True, "first")(x1)
    x2 = Concatenate()([x11, x12, x13, x14])

    x21 = Conv2dBatchNormMax(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_2", "first")(x2)
    x22 = SimpleResNet(16, 3, 1, "same", "srn_21", x2, kernel_initializer, kernel_regularizer, relu_position="first")(
        x2, x14)
    x23 = SimpleResNet(16, 3, 1, "same", "srn_22", x2, kernel_initializer, kernel_regularizer, relu_position="first")(
        x2, x1)
    x24 = SimpleResNet(16, 3, 1, "same", "srn_23", x2, kernel_initializer, kernel_regularizer, relu_position="first")(
        x2, x11)
    x3 = Concatenate()([x21, x22, x23, x24])

    x31 = SimpleResNet(16, 3, 1, "same", "srn_3", x3, kernel_initializer, kernel_regularizer, relu_position="first")(x3,x24)
    x32 = Conv2dBatchNormAvg(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_3", "first")(x3)
    x33 = Conv2dBatchNorm(16, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_3", "first")(x3)
    x34 = Conv2dBatchNormMax(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_3", "first")(x3)
    x4 = Concatenate()([x31, x32, x33, x34])

    x41 = Conv2dBatchNorm(16, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_4", "first")(x4)
    x42 = Conv2dBatchNormAvg(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_4", "first")(x4)
    x43 = Conv2dBatchNormMax(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_4", "first")(x4)
    x44 = SimpleResNet(16, 3, 1, "same", "srn_4", x4, kernel_initializer, kernel_regularizer, relu_position="first")(x4,x34)
    x5 = Concatenate()([x41, x42, x43, x44])

    x51 = SimpleResNet(16, 3, 1, "same", "srn_5", x5, kernel_initializer, kernel_regularizer, relu_position="first")(x5,x44)
    x52 = Conv2dBatchNormAvg(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_5", "first")(x5)
    x53 = Conv2dBatchNorm(16, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_5", "first")(x5)
    x54 = Conv2dBatchNormMax(16, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_5", "first")(x5)
    x6 = Concatenate()([x51, x52, x53, x54])

    x61 = SimpleResNet(32, 3, 2, "same", "srn_6", x6, kernel_initializer, kernel_regularizer, down_sample=True,
                       relu_position="first")(x6, x51)
    x62 = Conv2dBatchNormAvg(32, 3, 2, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_6", "first")(x6)
    x63 = Conv2dBatchNormMax(32, 3, 2, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_6", "first")(x6)
    x64 = Conv2dBatchNorm(32, 3, 2, "same", None, kernel_initializer, kernel_regularizer, "cb_6", "first")(x6)
    x7 = Concatenate()([x61, x62, x63, x64])

    x71 = SimpleResNet(32, 3, 1, "same", "srn_71", x7, kernel_initializer, kernel_regularizer, relu_position="first")(
        x7, x63)
    x72 = Conv2dBatchNorm(32, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_7", "first")(x7)
    x73 = Conv2dBatchNormMax(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_7", "first")(x7)
    x74 = Conv2dBatchNormAvg(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_7", "first")(x7)
    x8 = Concatenate()([x71, x72, x73, x74])

    x81 = SimpleResNet(32, 3, 1, "same", "srn_8", x8, kernel_initializer, kernel_regularizer, relu_position="first")(x8, x72)
    x82 = Conv2dBatchNorm(32, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_8", "first")(x8)
    x83 = Conv2dBatchNormAvg(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_8", "first")(x8)
    x84 = Conv2dBatchNormMax(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_8", "first")(x8)
    x9 = Concatenate()([x81, x82, x83, x84])

    x91 = SimpleResNet(32, 3, 1, "same", "srn_9", x9, kernel_initializer, kernel_regularizer, relu_position="first")(x9, x82)
    x92 = Conv2dBatchNorm(32, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_9", "first")(x9)
    x93 = Conv2dBatchNormAvg(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_9", "first")(x9)
    x94 = Conv2dBatchNormMax(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_9", "first")(x9)
    x10 = Concatenate()([x91, x92, x93, x94])

    x11 = SimpleResNet(32, 3, 1, "same", "srn_10", x10, kernel_initializer, kernel_regularizer, relu_position="first")(x10, x94)
    x12 = Conv2dBatchNormAvg(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_10", "first")(x10)
    x13 = Conv2dBatchNormMax(32, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_10", "first")(x10)
    x14 = Conv2dBatchNorm(32, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_10", "first")(x10)
    x1 = Concatenate()([x11, x12, x13, x14])

    x21 = SimpleResNet(64, 3, 2, "same", "srn_11", x1, kernel_initializer, kernel_regularizer, down_sample=True,
                       relu_position="first")(x1, x13)
    x22 = Conv2dBatchNormAvg(64, 3, 2, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_11", "first")(x1)
    x23 = Conv2dBatchNorm(64, 3, 2, "same", None, kernel_initializer, kernel_regularizer, "cb_11", "first")(x1)
    x24 = Conv2dBatchNormMax(64, 3, 2, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_11", "first")(x1)
    x2 = Concatenate()([x21, x22, x23, x24])

    x31 = Conv2dBatchNorm(64, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_12", "first")(x2)
    x32 = Conv2dBatchNormAvg(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_12", "first")(x2)
    x33 = Conv2dBatchNormMax(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_12", "first")(x2)
    x34 = SepConv(64, 3, 1, "same", kernel_initializer, kernel_regularizer, "sc_12", True, "first")(x2)
    x3 = Concatenate()([x31, x32, x33, x34])

    x41 = Conv2dBatchNorm(64, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_13", "first")(x3)
    x42 = Conv2dBatchNormAvg(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_13", "first")(x3)
    x43 = Conv2dBatchNormMax(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_13", "first")(x3)
    x44 = SimpleResNet(64, 3, 1, "same", "srn_13", x3, kernel_initializer, kernel_regularizer, relu_position="first")(
        x3, x34)
    x4 = Concatenate()([x41, x42, x43, x44])

    x51 = SepConv(64, 3, 1, "same", kernel_initializer, kernel_regularizer, "sc_14", True, "first")(x4)
    x52 = Conv2dBatchNorm(64, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_14", "first")(x4)
    x53 = Conv2dBatchNormMax(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_14", "first")(x4)
    x54 = Conv2dBatchNormAvg(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cba_14", "first")(x4)
    x5 = Concatenate()([x51, x52, x53, x54])

    x61 = SimpleResNet(64, 3, 1, "same", "srn_15_1", x5, kernel_initializer, kernel_regularizer, relu_position="first")(
        x5, x44)
    x62 = SimpleResNet(64, 3, 1, "same", "srn_15_2", x5, kernel_initializer, kernel_regularizer, relu_position="first")(
        x5, x54)
    x63 = Conv2dBatchNormMax(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cbm_15", "first")(x5)
    x64 = Conv2dBatchNorm(64, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_15", "first")(x5)
    x6 = Concatenate()([x61, x62, x63, x64])

    x = GlobalAveragePooling2D()(x6)
    x = Flatten()(x)
    x = tf.keras.layers.Dense(num_class, activation="softmax", name=f"classifier_1")(x)

    GLNASModel = tf.keras.models.Model(inputs=inputs, outputs=x)

    return GLNASModel


def createResNetFromGLNASManually(input_shape, num_class):
    inputs = tf.keras.layers.Input(shape=input_shape)

    kernel_initializer = tf.keras.initializers.he_normal()
    kernel_regularizer = tf.keras.regularizers.l2(1e-4)

    # x = Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, name="conv2d_0")(inputs)
    # x1 = BatchNormalization(name="batchnormalization_0")(x)
    # x = ReLU(name="relu_0")(x)
    # x1 = Conv2dBatchNorm(64, 3, 1, "same", None, kernel_initializer, kernel_regularizer, "cb_0")(inputs)
    x1 = StartingBackboneImgNet(64, kernel_initializer, kernel_regularizer, "sb_0")(inputs)

    # x = Conv2dBatchNormMax(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cb_1")(x1)
    #
    # # x = ReLU(name="relu_1")(x1)
    # # x = Conv2dBatchNorm(64, 3, 1, "same", None, "cb_2")(x)
    # x1 = SimpleResNet(64, 3, 1, "same", "srn_1", x, kernel_initializer, kernel_regularizer)(x, x1)
    #
    # x = Conv2dBatchNormMax(64, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cb_3")(x1)
    # # x = ReLU(name="relu_2")(x1)
    # # x = Conv2dBatchNorm(64, 3, 1, "same", None, "cb_4")(x)
    # x1 = SimpleResNet(64, 3, 1, "same", "srn_2", x, kernel_initializer, kernel_regularizer)(x, x1)
    # print(f"block 1: {x1}")
    #
    # x = Conv2dBatchNormMax(128, 3, 2, "same", None, 3, kernel_initializer, kernel_regularizer, "cb_5")(x1)
    # # x = ReLU(name="relu_3")(x1)
    # # x = Conv2dBatchNorm(128, 3, 1, "same", None, "cb_6")(x)
    # x1 = SimpleResNet(128, 3, 1, "same", "srn_3", x1, kernel_initializer, kernel_regularizer, down_sample=True, num_of_pool=1)(x, x1)
    # print(f"block 22: {x1}")
    #
    # x = Conv2dBatchNormMax(128, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cb_7")(x1)
    # # x = ReLU(name="relu_4")(x1)
    # # x = Conv2dBatchNorm(128, 3, 1, "same", None, "cb_8")(x)
    # x1 = SimpleResNet(128, 3, 1, "same", "srn_4", x, kernel_initializer, kernel_regularizer)(x, x1)
    # print(f"block 2: {x1}")
    #
    # x = Conv2dBatchNormMax(256, 3, 2, "same", None, 3, kernel_initializer, kernel_regularizer, "cb_9")(x1)
    # # x = ReLU(name="relu_5")(x1)
    # # x = Conv2dBatchNorm(256, 3, 1, "same", None, "cb_10")(x)
    # x1 = SimpleResNet(256, 3, 1, "same", "srn_5", x1, kernel_initializer, kernel_regularizer, down_sample=True, num_of_pool=1)(x, x1)
    #
    # x = Conv2dBatchNormMax(256, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cb_11")(x1)
    # # x = ReLU(name="relu_6")(x1)
    # # x = Conv2dBatchNorm(256, 3, 1, "same", None, "cb_12")(x)
    # x1 = SimpleResNet(256, 3, 1, "same", "srn_6", x, kernel_initializer, kernel_regularizer)(x, x1)
    # print(f"block 3: {x}")
    #
    # x = Conv2dBatchNormMax(512, 3, 2, "same", None, 3, kernel_initializer, kernel_regularizer, "cb_13")(x1)
    # # x = ReLU(name="relu_7")(x1)
    # # x = Conv2dBatchNorm(512, 3, 1, "same", None, "cb_14")(x)
    # x1 = SimpleResNet(512, 3, 1, "same", "srn_7", x1, kernel_initializer, kernel_regularizer, down_sample=True, num_of_pool=1)(x, x1)
    #
    # x = Conv2dBatchNormMax(512, 3, 1, "same", None, 3, kernel_initializer, kernel_regularizer, "cb_15")(x1)
    # # x = ReLU(name="relu_8")(x1)
    # # x = Conv2dBatchNorm(512, 3, 1, "same", None, "cb_16")(x)
    # x = SimpleResNet(512, 3, 1, "same", "srn_8", x, kernel_initializer, kernel_regularizer)(x, x1)
    # print(f"block 4: {x}")

    # x = Conv2dBatchNorm(64, 3, 1, "same", "sepconv_1")(x1)
    # x1 = SimpleResNet(64, 3, 1, "same", "srn_7", x1)(x, x1)
    # x = SepConv(64, 3, 1, "same", "sepconv_2")(x1)
    # x1 = SimpleResNet(64, 3, 1, "same", "srn_8", x1)(x, x1)
    #
    # x = SepConv(128, 3, 2, "same", "sepconv_5")(x1)
    # x1 = SimpleResNet(128, 3, 1, "same", "srn_1", x1, num_of_pool=1)(x, x1)
    # x = SepConv(128, 3, 1, "same", "sepconv_6")(x1)
    # x1 = SimpleResNet(128, 3, 1, "same", "srn_2", x1)(x, x1)
    #
    # x = SepConv(256, 3, 2, "same", "sepconv_7")(x1)
    # x1 = SimpleResNet(256, 3, 1, "same", "srn_3", x1, num_of_pool=1)(x, x1)
    # x = SepConv(256, 3, 1, "same", "sepconv_8")(x1)
    # x1 = SimpleResNet(256, 3, 1, "same", "srn_4", x1)(x, x1)
    #
    # x = SepConv(512, 3, 2, "same", "sepconv_9")(x1)
    # x1 = SimpleResNet(512, 3, 1, "same", "srn_5", x1, num_of_pool=1)(x, x1)
    # x = SepConv(512, 3, 1, "same", "sepconv_10")(x1)
    # x = SimpleResNet(512, 3, 1, "same", "srn_6", x1)(x, x1)

    # x = AveragePooling2D(pool_size=(4, 4))(x)
    x = GlobalAveragePooling2D()(x1)
    x = Flatten()(x)
    x = tf.keras.layers.Dense(num_class, activation="softmax", name=f"classifier_1")(x)

    GLNASModel = tf.keras.models.Model(inputs=inputs, outputs=x)

    return GLNASModel


