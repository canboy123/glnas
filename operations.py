import tensorflow as tf
from keras import backend as K
import math
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, \
AveragePooling2D, Add, ReLU, PReLU, LeakyReLU, DepthwiseConv2D, ELU, GlobalAveragePooling2D, SeparableConv2D

featureSearchSpace = [
    {'conv2d': {'filters': 32, "kernel": (3, 3), "stride": 1, "padding": "same", "activation": None}},
    {'conv2d': {'filters': 32, "kernel": (5, 5), "stride": 1, "padding": "same", "activation": None}},
    {'conv2d': {'filters': 32, "kernel": (7, 7), "stride": 1, "padding": "same", "activation": None}},
    {'conv2d': {'filters': 32, "kernel": (3, 3), "stride": 2, "padding": "valid", "activation": None}},
    {'conv2d': {'filters': 64, "kernel": (3, 3), "stride": 1, "padding": "same", "activation": None}},
    {'conv2d': {'filters': 64, "kernel": (5, 5), "stride": 1, "padding": "same", "activation": None}},
    {'conv2d': {'filters': 64, "kernel": (7, 7), "stride": 1, "padding": "same", "activation": None}},
    {'conv2d': {'filters': 64, "kernel": (3, 3), "stride": 2, "padding": "valid", "activation": None}},
    {'conv2d': {'filters': 128, "kernel": (3, 3), "stride": 1, "padding": "same", "activation": None}},
    {'conv2d': {'filters': 128, "kernel": (5, 5), "stride": 1, "padding": "same", "activation": None}},
    {'conv2d': {'filters': 128, "kernel": (7, 7), "stride": 1, "padding": "same", "activation": None}},
    {'conv2d': {'filters': 128, "kernel": (3, 3), "stride": 2, "padding": "valid", "activation": None}},
    {'depth2d': {"kernel": (3, 3), "stride": 1, "padding": "same", "activation": None}},
    {'depth2d': {"kernel": (5, 5), "stride": 1, "padding": "same", "activation": None}},
    {'depth2d': {"kernel": (7, 7), "stride": 1, "padding": "same", "activation": None}},
    {'depth2d': {"kernel": (3, 3), "stride": 2, "padding": "valid", "activation": None}},
    {'elu': {}},
    {'relu': {}},
    {'prelu': {}},
    {'leakyrelu': {}},
    {'batchnormalization': {}},
    {'averagegpool2d': {"pool": (3, 3), "stride": 1, "padding": "same"}},
    {'averagegpool2d': {"pool": (5, 5), "stride": 1, "padding": "same"}},
    {'averagegpool2d': {"pool": (7, 7), "stride": 1, "padding": "same"}},
    {'averagegpool2d': {"pool": (3, 3), "stride": 2, "padding": "valid"}},
    {'maxpooling2d': {"pool": (3, 3), "stride": 1, "padding": "same"}},
    {'maxpooling2d': {"pool": (5, 5), "stride": 1, "padding": "same"}},
    {'maxpooling2d': {"pool": (7, 7), "stride": 1, "padding": "same"}},
    {'maxpooling2d': {"pool": (3, 3), "stride": 2, "padding": "valid"}},
    {'globavgpooling2d': {"pool": (3, 3), "stride": 2, "padding": "valid"}},
    {'dropout': {"rate": 0.2}},
    {'dropout': {"rate": 0.4}},
    {'dropout': {"rate": 0.6}},
    {'dropout': {"rate": 0.8}},
]

featureBlockSearchSpace = [
    # {'conv2d': {'filters': 32, "kernel": (3, 3), "stride": 1, "padding": "same", "activation": None}},
    # {'conv2d': {'filters': 32, "kernel": (5, 5), "stride": 1, "padding": "same", "activation": None}},
    # {'conv2d': {'filters': 32, "kernel": (7, 7), "stride": 1, "padding": "same", "activation": None}},
    # {'conv2d': {'filters': 32, "kernel": (3, 3), "stride": 1, "padding": "valid", "activation": None}},
    # {'conv2d': {'filters': 32, "kernel": (3, 3), "stride": 2, "padding": "valid", "activation": None}},
    # {'conv2d': {'filters': 64, "kernel": (3, 3), "stride": 1, "padding": "same", "activation": None}},
    # {'conv2d': {'filters': 64, "kernel": (5, 5), "stride": 1, "padding": "same", "activation": None}},
    # {'conv2d': {'filters': 64, "kernel": (7, 7), "stride": 1, "padding": "same", "activation": None}},
    # {'conv2d': {'filters': 64, "kernel": (3, 3), "stride": 1, "padding": "valid", "activation": None}},
    # {'conv2d': {'filters': 64, "kernel": (3, 3), "stride": 2, "padding": "valid", "activation": None}},
    # {'conv2d': {'filters': 128, "kernel": (3, 3), "stride": 1, "padding": "same", "activation": None}}, # 10
    # {'conv2d': {'filters': 128, "kernel": (5, 5), "stride": 1, "padding": "same", "activation": None}},
    # {'conv2d': {'filters': 128, "kernel": (7, 7), "stride": 1, "padding": "same", "activation": None}},
    # {'conv2d': {'filters': 128, "kernel": (3, 3), "stride": 1, "padding": "valid", "activation": None}},
    # {'conv2d': {'filters': 128, "kernel": (3, 3), "stride": 2, "padding": "valid", "activation": None}},
    # {'depth2d': {"kernel": (3, 3), "stride": 1, "padding": "same", "activation": None}},
    # {'depth2d': {"kernel": (5, 5), "stride": 1, "padding": "same", "activation": None}},
    # {'depth2d': {"kernel": (7, 7), "stride": 1, "padding": "same", "activation": None}},
    # {'depth2d': {"kernel": (3, 3), "stride": 2, "padding": "valid", "activation": None}},
    # {'elu': {}},
    # {'relu': {}},   # 20
    # {'prelu': {}},
    # {'leakyrelu': {}},
    # {'batchnormalization': {}},
    # {'averagegpool2d': {"pool": (3, 3), "stride": 1, "padding": "same"}},
    # {'averagegpool2d': {"pool": (5, 5), "stride": 1, "padding": "same"}},
    # {'averagegpool2d': {"pool": (7, 7), "stride": 1, "padding": "same"}},
    # {'averagegpool2d': {"pool": (3, 3), "stride": 2, "padding": "valid"}},
    # {'maxpooling2d': {"pool": (3, 3), "stride": 1, "padding": "same"}},
    # {'maxpooling2d': {"pool": (5, 5), "stride": 1, "padding": "same"}},
    # {'maxpooling2d': {"pool": (7, 7), "stride": 1, "padding": "same"}}, # 30
    # {'maxpooling2d': {"pool": (3, 3), "stride": 2, "padding": "valid"}},
    # {'globavgpooling2d': {"pool": (3, 3), "stride": 1, "padding": "valid"}},
    # {'dropout': {"rate": 0.2}},
    # {'dropout': {"rate": 0.4}},
    # {'dropout': {"rate": 0.6}},
    # {'dropout': {"rate": 0.8}},
    {'convbatch': {'filters': 32, "kernel": (3, 3), "stride": 1, "padding": "same", "activation": None}},
    # {'convbatch': {'filters': 32, "kernel": (5, 5), "stride": 1, "padding": "same", "activation": None}},
    # {'convbatch': {'filters': 32, "kernel": (7, 7), "stride": 1, "padding": "same", "activation": None}},
    # {'convbatch': {'filters': 32, "kernel": (3, 3), "stride": 1, "padding": "valid", "activation": None}},  # 40
    # {'convbatch': {'filters': 32, "kernel": (3, 3), "stride": 2, "padding": "same", "activation": None}},
    # {'convbatch': {'filters': 32, "kernel": (3, 3), "stride": 2, "padding": "valid", "activation": None}},
    # {'convbatch': {'filters': 64, "kernel": (3, 3), "stride": 1, "padding": "same", "activation": None}},
    # {'convbatch': {'filters': 64, "kernel": (5, 5), "stride": 1, "padding": "same", "activation": None}},
    # {'convbatch': {'filters': 64, "kernel": (7, 7), "stride": 1, "padding": "same", "activation": None}},
    # {'convbatch': {'filters': 64, "kernel": (3, 3), "stride": 1, "padding": "valid", "activation": None}},
    # {'convbatch': {'filters': 64, "kernel": (3, 3), "stride": 2, "padding": "valid", "activation": None}},
    # {'convbatch': {'filters': 128, "kernel": (3, 3), "stride": 1, "padding": "same", "activation": None}},
    # {'convbatch': {'filters': 128, "kernel": (5, 5), "stride": 1, "padding": "same", "activation": None}},
    # {'convbatch': {'filters': 128, "kernel": (7, 7), "stride": 1, "padding": "same", "activation": None}},
    # {'convbatch': {'filters': 128, "kernel": (3, 3), "stride": 1, "padding": "valid", "activation": None}}, # 50
    # {'convbatch': {'filters': 128, "kernel": (3, 3), "stride": 2, "padding": "valid", "activation": None}},
    {'convbatchavg': {'filters': 32, "kernel": (3, 3), "stride": 1, "padding": "same", "activation": None, "pool": (3, 3)}},
    {'convbatchmax': {'filters': 32, "kernel": (3, 3), "stride": 1, "padding": "same", "activation": None, "pool": (3, 3)}},
    # {'convbatchmax': {'filters': 32, "kernel": (5, 5), "stride": 1, "padding": "same", "activation": None, "pool": (3, 3)}},
    # {'convbatchmax': {'filters': 32, "kernel": (7, 7), "stride": 1, "padding": "same", "activation": None, "pool": (3, 3)}},
    # {'convbatchmax': {'filters': 32, "kernel": (3, 3), "stride": 1, "padding": "valid", "activation": None, "pool": (3, 3)}},
    # {'convbatchmax': {'filters': 32, "kernel": (3, 3), "stride": 2, "padding": "valid", "activation": None, "pool": (3, 3)}},
    # {'convbatchmax': {'filters': 64, "kernel": (3, 3), "stride": 1, "padding": "same", "activation": None, "pool": (3, 3)}},
    # {'convbatchmax': {'filters': 64, "kernel": (5, 5), "stride": 1, "padding": "same", "activation": None, "pool": (3, 3)}},
    # {'convbatchmax': {'filters': 64, "kernel": (7, 7), "stride": 1, "padding": "same", "activation": None, "pool": (3, 3)}},
    # {'convbatchmax': {'filters': 64, "kernel": (3, 3), "stride": 1, "padding": "valid", "activation": None, "pool": (3, 3)}},   # 60
    # {'convbatchmax': {'filters': 64, "kernel": (3, 3), "stride": 2, "padding": "valid", "activation": None, "pool": (3, 3)}},
    # {'convbatchmax': {'filters': 128, "kernel": (3, 3), "stride": 1, "padding": "same", "activation": None, "pool": (3, 3)}},
    # {'convbatchmax': {'filters': 128, "kernel": (5, 5), "stride": 1, "padding": "same", "activation": None, "pool": (3, 3)}},
    # {'convbatchmax': {'filters': 128, "kernel": (7, 7), "stride": 1, "padding": "same", "activation": None, "pool": (3, 3)}},
    # {'convbatchmax': {'filters': 128, "kernel": (3, 3), "stride": 1, "padding": "valid", "activation": None, "pool": (3, 3)}},
    # {'convbatchmax': {'filters': 128, "kernel": (3, 3), "stride": 2, "padding": "valid", "activation": None, "pool": (3, 3)}},
    {'dilconv': {'filters': 32, "kernel": (3, 3), "stride": 1, "padding": "same"}},
    # {'dilconv': {'filters': 32, "kernel": (5, 5), "stride": 1, "padding": "same"}},
    # {'dilconv': {'filters': 32, "kernel": (7, 7), "stride": 1, "padding": "same"}},
    # {'dilconv': {'filters': 32, "kernel": (3, 3), "stride": 1, "padding": "valid"}},    # 70
    # {'dilconv': {'filters': 32, "kernel": (3, 3), "stride": 2, "padding": "same"}},
    # {'dilconv': {'filters': 32, "kernel": (3, 3), "stride": 2, "padding": "valid"}},
    # {'dilconv': {'filters': 64, "kernel": (3, 3), "stride": 1, "padding": "same"}},
    # {'dilconv': {'filters': 64, "kernel": (5, 5), "stride": 1, "padding": "same"}},
    # {'dilconv': {'filters': 64, "kernel": (7, 7), "stride": 1, "padding": "same"}},
    # {'dilconv': {'filters': 64, "kernel": (3, 3), "stride": 1, "padding": "valid"}},
    # {'dilconv': {'filters': 64, "kernel": (3, 3), "stride": 2, "padding": "valid"}},
    # {'dilconv': {'filters': 128, "kernel": (3, 3), "stride": 1, "padding": "same"}},
    # {'dilconv': {'filters': 128, "kernel": (5, 5), "stride": 1, "padding": "same"}},
    # {'dilconv': {'filters': 128, "kernel": (7, 7), "stride": 1, "padding": "same"}},
    # {'dilconv': {'filters': 128, "kernel": (3, 3), "stride": 1, "padding": "valid"}},   # 80
    # {'dilconv': {'filters': 128, "kernel": (3, 3), "stride": 2, "padding": "valid"}},
    {'sepconv': {'filters': 32, "kernel": (3, 3), "stride": 1, "padding": "same"}},
    # {'sepconv': {'filters': 32, "kernel": (5, 5), "stride": 1, "padding": "same"}},
    # {'sepconv': {'filters': 32, "kernel": (7, 7), "stride": 1, "padding": "same"}},
    # {'sepconv': {'filters': 32, "kernel": (3, 3), "stride": 1, "padding": "valid"}},
    # {'sepconv': {'filters': 32, "kernel": (3, 3), "stride": 2, "padding": "same"}},
    # {'sepconv': {'filters': 32, "kernel": (3, 3), "stride": 2, "padding": "valid"}},
    # {'sepconv': {'filters': 64, "kernel": (3, 3), "stride": 1, "padding": "same"}},
    # {'sepconv': {'filters': 64, "kernel": (5, 5), "stride": 1, "padding": "same"}},
    # {'sepconv': {'filters': 64, "kernel": (7, 7), "stride": 1, "padding": "same"}},
    # {'sepconv': {'filters': 64, "kernel": (3, 3), "stride": 1, "padding": "valid"}},    # 90
    # {'sepconv': {'filters': 64, "kernel": (3, 3), "stride": 2, "padding": "valid"}},
    # {'sepconv': {'filters': 128, "kernel": (3, 3), "stride": 1, "padding": "same"}},
    # {'sepconv': {'filters': 128, "kernel": (5, 5), "stride": 1, "padding": "same"}},
    # {'sepconv': {'filters': 128, "kernel": (7, 7), "stride": 1, "padding": "same"}},
    # {'sepconv': {'filters': 128, "kernel": (3, 3), "stride": 1, "padding": "valid"}},
    # {'sepconv': {'filters': 128, "kernel": (3, 3), "stride": 2, "padding": "valid"}},
]

classificationSearchSpace = [
    {'dense': {"unit": 128, "activation": None}},
    {'dense': {"unit": 128, "activation": "relu"}},
    {'dense': {"unit": 256, "activation": None}},
    {'dense': {"unit": 256, "activation": "relu"}},
    {'batchnormalization': {}},
    {'dropout': {"rate": 0.2}},
    {'dropout': {"rate": 0.4}},
    {'dropout': {"rate": 0.6}},
    {'dropout': {"rate": 0.8}},
]

# kernel_regularizer = tf.keras.regularizers.l2(1e-4)
OPS = {
    'conv2d': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: Conv2D(filters, kernel, stride, padding, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name=name),
    'depth2d': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: DepthwiseConv2D(kernel, stride, padding, activation=activation, depthwise_regularizer=kernel_regularizer, name=name),
    'averagegpool2d': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: AveragePooling2D(pool, stride, padding, name=name),
    'maxpooling2d': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: MaxPooling2D(pool, stride, padding, name=name),
    'globavgpooling2d': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: GlobalAveragePooling2D(name=name),
    'batchnormalization': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: BatchNormalization(name=name),
    'elu': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: ELU(name=name),
    'relu': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: ReLU(name=name),
    'prelu': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: PReLU(name=name),
    'leakyrelu': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: LeakyReLU(name=name),
    'dropout': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: Dropout(rate, name=name),
    'flatten': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: Flatten(name=name),
    'dense': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: Dense(unit, activation, kernel_regularizer=kernel_regularizer, name=name),
    'dilconv': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: DilConv(filters, kernel, stride, padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name=name, relu_position=relu_position),
    'sepconv': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: SepConv(filters, kernel, stride, padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name=name, relu_position=relu_position),
    'convbatch': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: Conv2dBatchNorm(filters, kernel, stride, padding, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name=name, relu_position=relu_position),
    'convdrop': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: Conv2dDropout(filters, kernel, stride, padding, activation=activation, rate=rate, name=name),
    'convbatchmax': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: Conv2dBatchNormMax(filters, kernel, stride, padding, activation=activation, pool=pool, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name=name, relu_position=relu_position),
    'convbatchavg': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: Conv2dBatchNormAvg(filters, kernel, stride, padding, activation=activation, pool=pool, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name=name, relu_position=relu_position),
    'convbatchmaxdrop': lambda filters, kernel, stride, activation, padding, pool, rate, unit, kernel_initializer, kernel_regularizer, name, relu_position: Conv2dBatchNormMaxDrop(filters, kernel, stride, padding, activation=activation, pool=pool, rate=rate, name=name),
}

class StartingBackboneImgNet(tf.keras.Model):
    def __init__(self, filters, kernel_initializer, kernel_regularizer, name, relu_position="last"):
        super(StartingBackboneImgNet, self).__init__()
        self.__filters = filters
        self.__kernel_initializer = kernel_initializer
        self.__kernel_regularizer = kernel_regularizer
        self._name = f"{name}"
        self.relu_position = relu_position

    def build(self, input_shape):
        if self.relu_position == "last":
            self.op = tf.keras.Sequential([
                            Conv2D(self.__filters, 3, 2, "same",
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d_1"),
                            BatchNormalization(name=f"{self._name}_batchNorm_1"),
                            ReLU(name=f"{self._name}_relu_1"),
                            Conv2D(self.__filters, 3, 2, "same",
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d_2"),
                            BatchNormalization(name=f"{self._name}_batchNorm_2"),
                            ReLU(name=f"{self._name}_relu_2"),
                            Conv2D(self.__filters, 3, 2, "same",
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d_3"),
                            BatchNormalization(name=f"{self._name}_batchNorm_3"),
                            ReLU(name=f"{self._name}_relu_3")
                        ])
        else:
            self.op = tf.keras.Sequential([
                            Conv2D(self.__filters, 3, 2, "same",
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d_1"),
                            BatchNormalization(name=f"{self._name}_batchNorm_1"),
                            ReLU(name=f"{self._name}_relu_1"),
                            Conv2D(self.__filters, 3, 2, "valid",
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d_2"),
                            BatchNormalization(name=f"{self._name}_batchNorm_2"),
                            ReLU(name=f"{self._name}_relu_3"),
                            Conv2D(self.__filters, 3, 2, "valid",
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d_3"),
                            BatchNormalization(name=f"{self._name}_batchNorm_3")
                        ])

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        x = self.op(inputs)

        return x

    # def get_config(self):
    #     return {"kernel_regularizer": self.__kernel_regularizer, "filters": self.__filters, "stride": self.__strides,
    #             "kernel": self.__kernel, "padding": self.__padding, "activation": self.__activation, "name": self._name}


class StartingBackbone(tf.keras.Model):
    def __init__(self, filters, kernel, stride, padding, activation, kernel_initializer, kernel_regularizer, name, relu_position="last"):
        super(StartingBackbone, self).__init__()
        self.__filters = filters
        self.__kernel = kernel
        self.__strides = stride
        self.__padding = padding
        self.__activation = activation
        self.__kernel_initializer = kernel_initializer
        self.__kernel_regularizer = kernel_regularizer
        self._name = f"{name}"
        self.relu_position = relu_position

    def build(self, input_shape):
        if self.relu_position == "last":
            self.op = tf.keras.Sequential([
                            Conv2D(self.__filters, 3, 1, "same",
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d_1"),
                            BatchNormalization(name=f"{self._name}_batchNorm_1"),
                            ReLU(name=f"{self._name}_relu_1"),
                            Conv2D(self.__filters, 1, 1, "valid",
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d_2"),
                            BatchNormalization(name=f"{self._name}_batchNorm_2"),
                            ReLU(name=f"{self._name}_relu_2")
                        ])
        else:
            self.op = tf.keras.Sequential([
                            Conv2D(self.__filters, 3, 1, "same",
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d_1"),
                            BatchNormalization(name=f"{self._name}_batchNorm_1"),
                            ReLU(name=f"{self._name}_relu_1"),
                            Conv2D(self.__filters, 1, 1, "valid",
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d_2"),
                            BatchNormalization(name=f"{self._name}_batchNorm_2")
                        ])

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        x = self.op(inputs)

        return x

    # def get_config(self):
    #     return {"kernel_regularizer": self.__kernel_regularizer, "filters": self.__filters, "stride": self.__strides,
    #             "kernel": self.__kernel, "padding": self.__padding, "activation": self.__activation, "name": self._name}

class Conv2dBatchNorm(tf.keras.Model):
    def __init__(self, filters, kernel, stride, padding, activation, kernel_initializer, kernel_regularizer, name, relu_position="last"):
        super(Conv2dBatchNorm, self).__init__()
        self.__filters = filters
        self.__kernel = kernel
        self.__strides = stride
        self.__padding = padding
        self.__activation = activation
        self.__kernel_initializer = kernel_initializer
        self.__kernel_regularizer = kernel_regularizer
        self._name = f"{name}"
        self.relu_position = relu_position

    def build(self, input_shape):
        if self.relu_position == "last":
            self.op = tf.keras.Sequential([
                            Conv2D(self.__filters, self.__kernel, self.__strides, self.__padding, activation=self.__activation,
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d"),
                            BatchNormalization(name=f"{self._name}_batchNorm"),
                            ReLU(name=f"{self._name}_relu")
                        ])
        else:
            self.op = tf.keras.Sequential([
                            ReLU(name=f"{self._name}_relu"),
                            Conv2D(self.__filters, self.__kernel, self.__strides, self.__padding, activation=self.__activation,
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d"),
                            BatchNormalization(name=f"{self._name}_batchNorm"),
                        ])

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        x = self.op(inputs)

        return x

    # def get_config(self):
    #     return {"kernel_regularizer": self.__kernel_regularizer, "filters": self.__filters, "stride": self.__strides,
    #             "kernel": self.__kernel, "padding": self.__padding, "activation": self.__activation, "name": self._name}

class Conv2dDropout(tf.keras.Model):
    def __init__(self, filters, kernel, stride, padding, activation, rate, name):
        super(Conv2dDropout, self).__init__()
        self._name = f"{name}_conv2ddropout"
        self._kernel_regularizer = tf.keras.regularizers.l2(2e-4)
        self.conv2d = Conv2D(filters, kernel, stride, padding, activation=activation,
                             kernel_regularizer=self._kernel_regularizer, name=f"{self._name}_conv2d")
        self.dropout = Dropout(rate, name=f"{self._name}_dropout")

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        x = self.conv2d(inputs)
        x = self.dropout(x)

        return x

class Conv2dBatchNormMax(tf.keras.Model):
    def __init__(self, filters, kernel, stride, padding, activation, pool, kernel_initializer, kernel_regularizer, name, relu_position="last"):
        super(Conv2dBatchNormMax, self).__init__()
        self.__filters = filters
        self.__kernel = kernel
        self.__strides = stride
        self.__padding = padding
        self.__activation = activation
        self.__pool = pool
        self.__kernel_initializer = kernel_initializer
        self.__kernel_regularizer = kernel_regularizer
        self._name = f"{name}"
        self.relu_position = relu_position

    def build(self, input_shape):
        if self.relu_position == "last":
            self.op = tf.keras.Sequential([
                            Conv2D(self.__filters, self.__kernel, self.__strides, self.__padding, activation=self.__activation,
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d"),
                            BatchNormalization(name=f"{self._name}_batchNorm"),
                            MaxPooling2D(self.__pool, 1, padding="same", name=f"{self._name}_maxpool"),
                            ReLU(name=f"{self._name}_relu"),
                        ])
        else:
            self.op = tf.keras.Sequential([
                            ReLU(name=f"{self._name}_relu"),
                            Conv2D(self.__filters, self.__kernel, self.__strides, self.__padding, activation=self.__activation,
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d"),
                            BatchNormalization(name=f"{self._name}_batchNorm"),
                            MaxPooling2D(self.__pool, 1, padding="same", name=f"{self._name}_maxpool"),
                        ])

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        x = self.op(inputs)

        return x

    # def get_config(self):
    #     return {"kernel_regularizer": self.__kernel_regularizer, "filters": self.__filters, "stride": self.__strides,
    #             "kernel": self.__kernel, "padding": self.__padding, "activation": self.__activation, "pool": self.__pool, "name": self._name}

class Conv2dBatchNormAvg(tf.keras.Model):
    def __init__(self, filters, kernel, stride, padding, activation, pool, kernel_initializer, kernel_regularizer, name, relu_position="last"):
        super(Conv2dBatchNormAvg, self).__init__()
        self.__filters = filters
        self.__kernel = kernel
        self.__strides = stride
        self.__padding = padding
        self.__activation = activation
        self.__pool = pool
        self.__kernel_initializer = kernel_initializer
        self.__kernel_regularizer = kernel_regularizer
        self._name = f"{name}"
        self.relu_position = relu_position

    def build(self, input_shape):
        if self.relu_position == "last":
            self.op = tf.keras.Sequential([
                            Conv2D(self.__filters, self.__kernel, self.__strides, self.__padding, activation=self.__activation,
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d"),
                            BatchNormalization(name=f"{self._name}_batchNorm"),
                            AveragePooling2D(self.__pool, 1, padding="same", name=f"{self._name}_avgpool"),
                            ReLU(name=f"{self._name}_relu"),
                        ])
        else:
            self.op = tf.keras.Sequential([
                            ReLU(name=f"{self._name}_relu"),
                            Conv2D(self.__filters, self.__kernel, self.__strides, self.__padding, activation=self.__activation,
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d"),
                            BatchNormalization(name=f"{self._name}_batchNorm"),
                            AveragePooling2D(self.__pool, 1, padding="same", name=f"{self._name}_avgpool"),
                        ])

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        x = self.op(inputs)

        return x

    # def get_config(self):
    #     return {"kernel_regularizer": self.__kernel_regularizer, "filters": self.__filters, "stride": self.__strides,
    #             "kernel": self.__kernel, "padding": self.__padding, "activation": self.__activation, "pool": self.__pool, "name": self._name}

class Conv2dBatchNormMaxDrop(tf.keras.Model):
    def __init__(self, filters, kernel, stride, padding, activation, pool, rate, name, kernel_regularizer=None, relu_position="last"):
        super(Conv2dBatchNormMaxDrop, self).__init__()
        self.__filters = filters
        self.__strides = stride
        self.__kernel = kernel
        self.__padding = padding
        self.__activation = activation
        self.__pool = pool
        self.__rate = rate
        self._name = f"{name}"
        self.__kernel_regularizer = kernel_regularizer
        # self._kernel_regularizer = tf.keras.regularizers.l2(2e-4)
        self.relu_position = relu_position

    def build(self, input_shape):
        self.conv2d = Conv2D(self.__filters, self.__kernel, self.__strides, self.__padding, activation=self.__activation,
                             kernel_regularizer=self.__kernel_regularizer, name=f"{self._name}_conv2d")
        self.batchNorm = BatchNormalization(name=f"{self._name}_batchNorm")
        self.maxPooling = MaxPooling2D(self.__pool, self.__strides, padding=self.__padding, name=f"{self._name}_maxpool")
        self.dropout = Dropout(self.__rate, name=f"{self._name}_dropout")


    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        x = self.conv2d(inputs)
        x = self.batchNorm(x)
        x = self.maxPooling(x)
        x = self.dropout(x)

        return x

    # def get_config(self):
    #     return {"kernel_regularizer": self.__kernel_regularizer, "filters": self.__filters, "stride": self.__strides,
    #             "kernel": self.__kernel, "padding": self.__padding, "activation": self.__activation, "pool": self.__pool,
    #             "rate": self.__rate, "name": self._name}

class SimpleResNet(tf.keras.Model):
    def __init__(self, filter, kernel, stride, padding, name, pre_layer, kernel_initializer=None, kernel_regularizer=None,
                 pre_layer_setting=None, down_sample=False, num_of_pool=0, relu_position="last"):
        super().__init__()

        self.__filter = filter
        self.__kernel_initializer = kernel_initializer
        self.__kernel_regularizer = kernel_regularizer
        self.__down_sample = down_sample
        self.__num_of_pool = num_of_pool
        self.__strides = stride
        # self.__kernel = kernel
        # self.__padding = padding
        self.__pre_layer = pre_layer
        self.__pre_layer_shape = pre_layer.shape
        # self.__pre_layer_setting = pre_layer_setting
        self._name = f"{name}"
        self.relu_position = relu_position

    def build(self, input_shape):
        if self.relu_position == "last":
            self.op = tf.keras.Sequential([
                            Conv2D(self.__filter, kernel_size=3, strides=self.__strides, padding='same', use_bias=False,
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer,
                                   name=f"{self._name}_conv_1"),
                            BatchNormalization(name=f"{self._name}_bn1"),
                        ])
        else:
            self.op = tf.keras.Sequential([
                            ReLU(name=f"{self._name}_relu_1"),
                            Conv2D(self.__filter, kernel_size=3, strides=self.__strides, padding='same', use_bias=False,
                                   kernel_initializer=self.__kernel_initializer,
                                   kernel_regularizer=self.__kernel_regularizer,
                                   name=f"{self._name}_conv_1"),
                            BatchNormalization(name=f"{self._name}_bn1"),
                        ])

        if self.__down_sample:
            if self.relu_position == "last":
                self.shortcut = tf.keras.Sequential([
                                    Conv2D(self.__filter, kernel_size=1, strides=self.__strides, use_bias=False,
                                           kernel_initializer=self.__kernel_initializer,
                                           kernel_regularizer=self.__kernel_regularizer,
                                           name=f"{self._name}_conv_2"),
                                    BatchNormalization(name=f"{self._name}_bn2")
                                ])
            else:
                self.shortcut = tf.keras.Sequential([
                                    ReLU(name=f"{self._name}_relu_2"),
                                    Conv2D(self.__filter, kernel_size=1, strides=self.__strides, use_bias=False,
                                           kernel_initializer=self.__kernel_initializer,
                                           kernel_regularizer=self.__kernel_regularizer,
                                           name=f"{self._name}_conv_2"),
                                    BatchNormalization(name=f"{self._name}_bn2")
                                ])
        else:
            if self.__pre_layer.shape[-1] == self.__filter and input_shape[-1] == self.__filter:
                self.shortcut = lambda x: x
            else:
                if self.relu_position == "last":
                    self.shortcut = tf.keras.Sequential([
                                        Conv2D(self.__filter, kernel_size=1, strides=self.__strides, use_bias=False,
                                               kernel_initializer=self.__kernel_initializer,
                                               kernel_regularizer=self.__kernel_regularizer,
                                               name=f"{self._name}_conv_2"),
                                        BatchNormalization(name=f"{self._name}_bn2")
                                    ])
                else:
                    self.shortcut = tf.keras.Sequential([
                                        ReLU(name=f"{self._name}_relu_2"),
                                        Conv2D(self.__filter, kernel_size=1, strides=self.__strides, use_bias=False,
                                               kernel_initializer=self.__kernel_initializer,
                                               kernel_regularizer=self.__kernel_regularizer,
                                               name=f"{self._name}_conv_2"),
                                        BatchNormalization(name=f"{self._name}_bn2")
                                    ])

        if self.__num_of_pool > 0:
            self.tmp_model = tf.keras.Sequential()
            if self.relu_position == "first":
                self.tmp_model.add(ReLU(name=f"{self._name}_bn3"))
            for i in range(self.__num_of_pool):
                tmp_filter = input_shape[-1] / 2 ** (self.__num_of_pool-i-1)
                self.tmp_model.add(Conv2D(tmp_filter, kernel_size=1, strides=2, use_bias=False,
                                          kernel_initializer=self.__kernel_initializer,
                                          kernel_regularizer=self.__kernel_regularizer,
                                          name=f"{self._name}_conv_{i + 4}"))
                self.tmp_model.add(BatchNormalization(name=f"{self._name}_bn{i + 4}"))
        else:
            self.tmp_model = lambda x: x

        self.merge = Add()

    def call(self, input, input2):
        # print("input", input)
        # print("input2", input2)
        input2 = self.tmp_model(input2)
        # print("after input2 1", input2)
        input2 = self.shortcut(input2)
        # print("after input2 2", input2)

        # x = tf.keras.activations.relu(self.bn1(self.conv1(input)))
        x = self.op(input)
        # print("after input", x)
        # x = self.merge([self.shortcut(input2), x])
        x = self.merge([input2, x])
        x = tf.keras.activations.relu(x)

        return x

    # def get_config(self):
    #     config = super(SimpleResNet, self).get_config()
    #     config.update({"kernel": self.__kernel, "stride": self.__strides, "padding": self.__padding, "name": self._name,
    #             "pre_layer": self.__pre_layer, "kernel_regularizer": self.__kernel_regularizer,
    #             "pre_layer_setting": self.__pre_layer_setting,
    #             "down_sample": self.__down_sample})
    #     return config

    # def get_config(self):
    #     return {"kernel": self.__kernel, "stride": self.__strides, "padding": self.__padding, "name": self._name,
    #             "pre_layer": self.__pre_layer, "kernel_regularizer": self.__kernel_regularizer,
    #             "pre_layer_setting": self.__pre_layer_setting,
    #             "down_sample": self.__down_sample}
    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

class DilConv(tf.keras.Model):
    def __init__(self, filters, kernel, stride, padding, kernel_initializer, kernel_regularizer, name, affine=True, relu_position="last"):
        super(DilConv, self).__init__()
        self.__filters = filters
        self.__kernel = kernel
        self.__stride = stride
        self.__padding = padding
        self.__kernel_initializer = kernel_initializer
        self.__kernel_regularizer = kernel_regularizer
        self._name = name
        self._affine = affine
        self.relu_position = relu_position

    def build(self, input_shape):
        if self.relu_position == "first":
            self.op = tf.keras.Sequential([
                            SeparableConv2D(filters=self.__filters, kernel_size=self.__kernel, strides=self.__stride,
                                            depthwise_initializer=self.__kernel_initializer,
                                            pointwise_initializer=self.__kernel_initializer,
                                            depthwise_regularizer=self.__kernel_regularizer,
                                            pointwise_regularizer=self.__kernel_regularizer,
                                            padding='same', use_bias=False, name=f"{self._name}_conv2d"),
                            BatchNormalization(name=f"{self._name}_batchNorm"),
                            ReLU(name=f"{self._name}_relu")
                        ])
        else:
            self.op = tf.keras.Sequential([
                            ReLU(name=f"{self._name}_relu"),
                            SeparableConv2D(filters=self.__filters, kernel_size=self.__kernel, strides=self.__stride,
                                            depthwise_initializer=self.__kernel_initializer,
                                            pointwise_initializer=self.__kernel_initializer,
                                            depthwise_regularizer=self.__kernel_regularizer,
                                            pointwise_regularizer=self.__kernel_regularizer,
                                            padding='same', use_bias=False, name=f"{self._name}_conv2d"),
                            BatchNormalization(name=f"{self._name}_batchNorm"),
                        ])

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        x = self.op(inputs)

        return x

    def get_config(self):
        return {"filters": self.__filters, "stride": self.__stride,
                "kernel": self.__kernel, "padding": self.__padding,
                "kernel_initializer": self.__kernel_initializer,
                "kernel_regularizer": self.__kernel_regularizer,
                "name": self._name}


class SepConv(tf.keras.Model):
    def __init__(self, filters, kernel, stride, padding, kernel_initializer, kernel_regularizer, name, affine=True, relu_position="last"):
        super(SepConv, self).__init__()
        self.__filters = filters
        self.__kernel = kernel
        self.__stride = stride
        self.__padding = padding
        self.__kernel_initializer = kernel_initializer
        self.__kernel_regularizer = kernel_regularizer
        self._name = name
        self._affine = affine
        self.relu_position = relu_position


    def build(self, input_shape):
        if self.relu_position == "first":
            self.op = tf.keras.Sequential([
                            SeparableConv2D(filters=self.__filters, kernel_size=self.__kernel, strides=self.__stride, padding='same',
                                            depthwise_initializer=self.__kernel_initializer,
                                            pointwise_initializer=self.__kernel_initializer,
                                            depthwise_regularizer=self.__kernel_regularizer,
                                            pointwise_regularizer=self.__kernel_regularizer,
                                            use_bias=False, name=f"{self._name}_conv2d_1"),
                            BatchNormalization(name=f"{self._name}_batchNorm_1"),
                            ReLU(name=f"{self._name}_relu_1"),
                            SeparableConv2D(filters=self.__filters, kernel_size=self.__kernel, strides=1, padding='same',
                                            depthwise_initializer=self.__kernel_initializer,
                                            pointwise_initializer=self.__kernel_initializer,
                                            depthwise_regularizer=self.__kernel_regularizer,
                                            pointwise_regularizer=self.__kernel_regularizer,
                                            use_bias=False, name=f"{self._name}_conv2d_2"),
                            BatchNormalization(name=f"{self._name}_batchNorm_2"),
                            ReLU(name=f"{self._name}_relu_2")
                        ])
        else:
            self.op = tf.keras.Sequential([
                            ReLU(name=f"{self._name}_relu_1"),
                            SeparableConv2D(filters=self.__filters, kernel_size=self.__kernel, strides=self.__stride, padding='same',
                                            depthwise_initializer=self.__kernel_initializer,
                                            pointwise_initializer=self.__kernel_initializer,
                                            depthwise_regularizer=self.__kernel_regularizer,
                                            pointwise_regularizer=self.__kernel_regularizer,
                                            use_bias=False, name=f"{self._name}_conv2d_1"),
                            BatchNormalization(name=f"{self._name}_batchNorm_1"),
                            ReLU(name=f"{self._name}_relu_2"),
                            SeparableConv2D(filters=self.__filters, kernel_size=self.__kernel, strides=1, padding='same',
                                            depthwise_initializer=self.__kernel_initializer,
                                            pointwise_initializer=self.__kernel_initializer,
                                            depthwise_regularizer=self.__kernel_regularizer,
                                            pointwise_regularizer=self.__kernel_regularizer,
                                            use_bias=False, name=f"{self._name}_conv2d_2"),
                            BatchNormalization(name=f"{self._name}_batchNorm_2"),
                        ])

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        x = self.op(inputs)

        return x

    def get_config(self):
        return {"filters": self.__filters, "stride": self.__stride,
                "kernel": self.__kernel, "padding": self.__padding,
                "kernel_initializer": self.__kernel_initializer,
                "kernel_regularizer": self.__kernel_regularizer,
                "name": self._name}