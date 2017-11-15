import math

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import optimizers
from tensorflow.contrib.keras.python.keras import regularizers
from tensorflow.contrib.keras.python.keras.initializers import he_normal
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, \
    Input, add, GlobalAveragePooling2D, AveragePooling2D, Lambda, SeparableConv2D, GlobalMaxPooling2D, concatenate, \
    Reshape, multiply
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


def get_LeNet(cfg):
    load_model = cfg['load_model']
    input_shape = cfg['input_shape']
    num_classes = cfg['num_classes']

    model = Sequential()
    model.add(
        Conv2D(6, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(84, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))
    model.summary()

    if load_model:
        model.load_weights('lenet.h5', by_name=True)

    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def get_Vgg16(cfg):
    load_model = cfg['load_model']
    dropout = cfg['dropout']
    weight_decay = cfg['weight_decay']
    num_classes = cfg['num_classes']
    WEIGHTS_PATH = cfg['WEIGHTS_PATH']
    filepath = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')
    input_shape = cfg['input_shape']

    model = Sequential()
    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block1_conv1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block3_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block4_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block5_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block5_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
                    kernel_initializer=he_normal(), name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4096, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                    name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, kernel_regularizer=keras.regularizers.l2(weight_decay),
                    kernel_initializer=he_normal(), name='predictions_fc'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    model.summary()

    if not load_model:
        model.load_weights(filepath, by_name=True)
    else:
        model.load_weights('vgg16.h5', by_name=True)

    return model


def get_Vgg19(cfg):
    load_model = cfg['load_model']
    dropout = cfg['dropout']
    weight_decay = cfg['weight_decay']
    num_classes = cfg['num_classes']
    WEIGHTS_PATH = cfg['WEIGHTS_PATH']
    filepath = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')
    input_shape = cfg['input_shape']

    # build model
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block1_conv1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block3_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block3_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block4_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block4_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block5_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block5_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                     kernel_initializer=he_normal(), name='block5_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
                    kernel_initializer=he_normal(), name='fc'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4096, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                    name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, kernel_regularizer=keras.regularizers.l2(weight_decay),
                    kernel_initializer=he_normal(), name='predictions_fc'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    model.summary()

    if not load_model:
        model.load_weights(filepath, by_name=True)
    else:
        model.load_weights('vgg19.h5', by_name=True)

    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def get_ResNet(cfg):
    load_model = cfg['load_model']
    weight_decay = cfg['weight_decay']
    num_classes = cfg['num_classes']
    input_shape = cfg['input_shape']
    stack_n = cfg['stack_n']

    def residual_block(x, shape, increase_filter=False):
        output_filter_num = shape[1]
        if increase_filter:
            first_stride = (2, 2)
        else:
            first_stride = (1, 1)

        pre_bn = BatchNormalization()(x)
        pre_relu = Activation('relu')(pre_bn)

        conv_1 = Conv2D(output_filter_num,
                        kernel_size=(3, 3),
                        strides=first_stride,
                        padding='same',
                        kernel_initializer=he_normal(),
                        kernel_regularizer=regularizers.l2(weight_decay)
                        )(pre_relu)
        bn_1 = BatchNormalization()(conv_1)
        relu1 = Activation('relu')(bn_1)
        conv_2 = Conv2D(output_filter_num,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=he_normal(),
                        kernel_regularizer=regularizers.l2(weight_decay)
                        )(relu1)
        if increase_filter:
            projection = Conv2D(output_filter_num,
                                kernel_size=(1, 1),
                                strides=(2, 2),
                                padding='same',
                                kernel_initializer=he_normal(),
                                kernel_regularizer=regularizers.l2(weight_decay)
                                )(x)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    img_input = Input(shape=input_shape)
    x = Conv2D(filters=16,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               kernel_initializer=he_normal(),
               kernel_regularizer=regularizers.l2(weight_decay)
               )(img_input)

    for _ in range(0, stack_n):
        x = residual_block(x, [16, 16])

    x = residual_block(x, [16, 32], increase_filter=True)
    for _ in range(1, stack_n):
        x = residual_block(x, [16, 32])

    x = residual_block(x, [32, 64], increase_filter=True)
    for _ in range(1, stack_n):
        x = residual_block(x, [32, 64])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax', kernel_initializer=he_normal(),
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    model = Model(img_input, x)
    model.summary()

    if load_model:
        model.load_weights('resnet.h5', by_name=True)

    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def get_DenseNet(cfg):
    load_model = cfg['load_model']
    weight_decay = cfg['weight_decay']
    num_classes = cfg['num_classes']
    input_shape = cfg['input_shape']

    growth_rate = cfg['growth_rate ']
    depth = cfg['depth']
    compression = cfg['compression']

    def bn_relu(x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def bottleneck(x):
        channels = growth_rate * 4
        x = bn_relu(x)
        x = Conv2D(channels, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(x)
        x = bn_relu(x)
        x = Conv2D(growth_rate, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(x)
        return x

    def single(x):
        x = bn_relu(x)
        x = Conv2D(growth_rate, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(x)
        return x

    def transition(x, inchannels):
        x = bn_relu(x)
        x = Conv2D(int(inchannels * compression), kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay),
                   use_bias=False)(x)
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
        return x

    def dense_block(x, blocks, nchannels):
        concat = x
        for i in range(blocks):
            x = bottleneck(concat)
            concat = concatenate([x, concat], axis=-1)
            nchannels += growth_rate
        return concat, nchannels

    def dense_layer(x):
        return Dense(num_classes, activation='softmax', kernel_initializer=he_normal(),
                     kernel_regularizer=regularizers.l2(weight_decay))(x)

    nblocks = (depth - 4) // 6
    nchannels = growth_rate * 2

    img_input = Input(shape=input_shape)
    x = Conv2D(nchannels, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=he_normal(),
               kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(img_input)

    x, nchannels = dense_block(x, nblocks, nchannels)
    x = transition(x, nchannels)
    x, nchannels = dense_block(x, nblocks, nchannels)
    x = transition(x, nchannels)
    x, nchannels = dense_block(x, nblocks, nchannels)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)

    model = Model(img_input, x)
    model.summary()

    if load_model:
        model.load_weights('densenet.h5', by_name=True)

    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


def get_ResNext(cfg):
    load_model = cfg['load_model']
    weight_decay = cfg['weight_decay']
    num_classes = cfg['num_classes']
    input_shape = cfg['input_shape']

    cardinality = cfg['cardinality']  # 4 or 8 or 16 or 32
    base_width = cfg['base_width']
    inplanes = cfg['inplanes']
    expansion = cfg['expansion']

    def add_common_layer(x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def group_conv(x, planes, stride):
        h = planes // cardinality
        groups = []
        for i in range(cardinality):
            group = Lambda(lambda z: z[:, :, :, i * h: i * h + h])(x)
            groups.append(Conv2D(h, kernel_size=(3, 3), strides=stride, kernel_initializer=he_normal(),
                                 kernel_regularizer=regularizers.l2(weight_decay), padding='same',
                                 use_bias=False)(
                group))
        x = concatenate(groups)
        return x

    def residual_block(x, planes, stride=(1, 1)):
        D = int(math.floor(planes * (base_width / 64.0)))
        C = cardinality

        shortcut = x

        y = Conv2D(D * C, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(shortcut)
        y = add_common_layer(y)

        y = group_conv(y, D * C, stride)
        y = add_common_layer(y)

        y = Conv2D(planes * expansion, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay),
                   use_bias=False)(y)
        y = add_common_layer(y)

        if stride != (1, 1) or inplanes != planes * expansion:
            shortcut = Conv2D(planes * expansion, kernel_size=(1, 1), strides=stride, padding='same',
                              kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay),
                              use_bias=False)(x)
            shortcut = BatchNormalization()(shortcut)
        y = add([y, shortcut])
        y = Activation('relu')(y)
        return y

    def residual_layer(x, blocks, planes, stride=(1, 1)):
        x = residual_block(x, planes, stride)
        nonlocal inplanes
        inplanes = planes * expansion
        for i in range(1, blocks):
            x = residual_block(x, planes)
        return x

    def conv3x3(x, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(x)
        return add_common_layer(x)

    def dense_layer(x):
        return Dense(num_classes, activation='softmax', kernel_initializer=he_normal(),
                     kernel_regularizer=regularizers.l2(weight_decay))(x)

    img_input = Input(shape=input_shape)
    # build the resnext model
    x = conv3x3(img_input, 64)
    x = residual_layer(x, 3, 64)
    x = residual_layer(x, 3, 128, stride=(2, 2))
    x = residual_layer(x, 3, 256, stride=(2, 2))
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)

    model = Model(img_input, x)
    model.summary()
    if load_model:
        model.load_weights('resnext.h5', by_name=True)
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def get_Xception(cfg):
    load_model = cfg['load_model']
    weight_decay = cfg['weight_decay']
    num_classes = cfg['num_classes']
    input_shape = cfg['input_shape']
    WEIGHTS_PATH = cfg['WEIGHTS_PATH']

    img_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    x = GlobalMaxPooling2D()(x)

    x = Dense(num_classes, activation='softmax', kernel_initializer=he_normal(),
              kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = Model(img_input, x, name='xception')
    model.summary()
    if not load_model:
        weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH, cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
    else:
        model.load_weights('xception.h5')

    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def get_InceptionV3(cfg):
    load_model = cfg['load_model']
    num_classes = cfg['num_classes']
    input_shape = cfg['input_shape']

    def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1)):
        x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization(scale=False)(x)
        x = Activation('relu')(x)
        return x

    img_input = Input(shape=input_shape)

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate([branch3x3, branch7x7x3, branch_pool], axis=-1, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = concatenate([branch3x3_1, branch3x3_2], axis=-1, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=-1)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='mixed' + str(9 + i))

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(img_input, x, name='inception_v3')
    model.summary()

    if load_model:
        model.load_weights('inceptionv3.h5')

    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def get_SENet(cfg):
    load_model = cfg['load_model']
    weight_decay = cfg['weight_decay']
    num_classes = cfg['num_classes']
    input_shape = cfg['input_shape']

    cardinality = cfg['cardinality']  # 4 or 8 or 16 or 32
    base_width = cfg['base_width']
    inplanes = cfg['inplanes']
    expansion = cfg['expansion']

    def add_common_layer(x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def group_conv(x, planes, stride):
        h = planes // cardinality
        groups = []
        for i in range(cardinality):
            group = Lambda(lambda z: z[:, :, :, i * h: i * h + h])(x)
            groups.append(Conv2D(h, kernel_size=(3, 3), strides=stride, kernel_initializer=he_normal(),
                                 kernel_regularizer=regularizers.l2(weight_decay), padding='same', use_bias=False)(
                group))
        x = concatenate(groups)
        return x

    def residual_block(x, planes, stride=(1, 1)):
        D = int(math.floor(planes * (base_width / 64.0)))
        C = cardinality

        shortcut = x

        y = Conv2D(D * C, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(shortcut)
        y = add_common_layer(y)

        y = group_conv(y, D * C, stride)
        y = add_common_layer(y)

        y = Conv2D(planes * expansion, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(y)
        y = add_common_layer(y)

        if stride != (1, 1) or inplanes != planes * expansion:
            shortcut = Conv2D(planes * expansion, kernel_size=(1, 1), strides=stride, padding='same',
                              kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay),
                              use_bias=False)(x)
            shortcut = BatchNormalization()(shortcut)

        y = squeeze_excite_block(y)
        y = add([y, shortcut])
        y = Activation('relu')(y)
        return y

    def residual_layer(x, blocks, planes, stride=(1, 1)):
        x = residual_block(x, planes, stride)
        nonlocal inplanes
        inplanes = planes * expansion
        for i in range(1, blocks):
            x = residual_block(x, planes)
        return x

    def squeeze_excite_block(input, ratio=16):
        init = input
        filters = init._keras_shape[-1]  # infer input number of filters
        se_shape = (1, 1, filters)
        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        x = multiply([init, se])
        return x

    def conv3x3(x, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(x)
        return add_common_layer(x)

    def dense_layer(x):
        return Dense(num_classes, activation='softmax', kernel_initializer=he_normal(),
                     kernel_regularizer=regularizers.l2(weight_decay))(x)

    img_input = Input(shape=input_shape)
    # build the resnext model
    x = conv3x3(img_input, 64)
    x = residual_layer(x, 3, 64)
    x = residual_layer(x, 3, 128, stride=(2, 2))
    x = residual_layer(x, 3, 256, stride=(2, 2))
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x


MODELS = {'LeNet': get_LeNet,
          'Vgg16': get_Vgg16,
          'Vgg19': get_Vgg19,
          'ResNet': get_ResNet,
          'DenseNet': get_DenseNet,
          'ResNext': get_ResNext,
          'Xception': get_Xception,
          'InceptionV3': get_InceptionV3,
          }
