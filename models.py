import math

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import optimizers
from tensorflow.contrib.keras.python.keras import regularizers
from tensorflow.contrib.keras.python.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.contrib.keras.python.keras.initializers import he_normal
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, \
    Input, add, GlobalAveragePooling2D, AveragePooling2D, Lambda, SeparableConv2D, GlobalMaxPooling2D, concatenate
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


class NNModel:
    def set_data(self, data):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class Vgg19(NNModel):
    def __init__(self, data, load_model=False):
        dropout = 0.5
        weight_decay = 0.0005
        self.num_classes = 1311
        WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
        filepath = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')

        self.data = self.set_data(data)
        (x_train, y_train), (x_test, y_test) = self.data

        # build model
        self.model = Sequential()

        # Block 1
        self.model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block1_conv1', input_shape=x_train.shape[1:]))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block1_conv2'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        self.model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block2_conv1'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block2_conv2'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        self.model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block3_conv1'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block3_conv2'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block3_conv3'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block3_conv4'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block4_conv1'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block4_conv2'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block4_conv3'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block4_conv4'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block5_conv1'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block5_conv2'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block5_conv3'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                              kernel_initializer=he_normal(), name='block5_conv4'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(Flatten(name='flatten'))
        self.model.add(Dense(4096, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
                             kernel_initializer=he_normal(), name='fc'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(
            Dense(4096, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                  name='fc2'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(
            Dense(self.num_classes, kernel_regularizer=keras.regularizers.l2(weight_decay),
                  kernel_initializer=he_normal(),
                  name='predictions_fc'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('softmax'))
        self.model.summary()

        if not load_model:
            self.model.load_weights(filepath, by_name=True)
        else:
            self.model.load_weights('vgg19.h5', by_name=True)

        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def set_data(self, data):
        (x_train, y_train), (x_test, y_test) = data
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # data preprocessing
        x_train[:, :, :, 0] = (x_train[:, :, :, 0] - 123.680)
        x_train[:, :, :, 1] = (x_train[:, :, :, 1] - 116.779)
        x_train[:, :, :, 2] = (x_train[:, :, :, 2] - 103.939)
        x_test[:, :, :, 0] = (x_test[:, :, :, 0] - 123.680)
        x_test[:, :, :, 1] = (x_test[:, :, :, 1] - 116.779)
        x_test[:, :, :, 2] = (x_test[:, :, :, 2] - 103.939)

        return (x_train, y_train), (x_test, y_test)

    def train(self):
        (x_train, y_train), (x_test, y_test) = self.data
        batch_size = 128
        epochs = 200
        iterations = 391
        log_filepath = r'./vgg19/'

        def scheduler(epoch):
            if epoch <= 60:
                return 0.05
            if epoch <= 120:
                return 0.01
            if epoch <= 160:
                return 0.002
            return 0.0004

        tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
        change_lr = LearningRateScheduler(scheduler)
        cbks = [change_lr, tb_cb]

        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True,
                                     width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant', cval=0.)

        datagen.fit(x_train)
        self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                 steps_per_epoch=iterations,
                                 epochs=epochs,
                                 callbacks=cbks,
                                 validation_data=(x_test, y_test))
        self.model.save('vgg19.h5')

    def evaluate(self):
        (x_train, y_train), (x_test, y_test) = self.data
        loss, acc = self.model.evaluate(x=x_test[:512, :, :, :], y=y_test[:512], batch_size=64, verbose=1)
        print('test_loss:', loss, 'test_acc:', acc)


class ResNet(NNModel):
    def __init__(self, data, load_model=False):
        self.classes_num = 1311
        img_rows, img_cols = 31, 31
        img_channels = 3
        stack_n = 18
        weight_decay = 0.0005
        self.data = self.set_data(data)

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

        img_input = Input(shape=(img_rows, img_cols, img_channels))
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
        x = Dense(self.classes_num, activation='softmax', kernel_initializer=he_normal(),
                  kernel_regularizer=regularizers.l2(weight_decay))(x)
        self.model = Model(img_input, x)
        self.model.summary()

        if load_model:
            self.model.load_weights('resnet.h5', by_name=True)

        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def set_data(self, data):
        (x_train, y_train), (x_test, y_test) = data
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
            x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]

        y_train = keras.utils.to_categorical(y_train, self.classes_num)
        y_test = keras.utils.to_categorical(y_test, self.classes_num)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        return (x_train, y_train), (x_test, y_test)

    def train(self):
        batch_size = 128
        epochs = 200
        iterations = 391
        log_filepath = r'./resnet50/'

        def scheduler(epoch):
            if epoch <= 60:
                return 0.05
            if epoch <= 120:
                return 0.01
            if epoch <= 160:
                return 0.002
            return 0.0004

        (x_train, y_train), (x_test, y_test) = self.data
        tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
        change_lr = LearningRateScheduler(scheduler)
        cbks = [change_lr, tb_cb]

        # set data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True,
                                     width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant', cval=0.)

        datagen.fit(x_train)

        # start training
        self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                 steps_per_epoch=iterations,
                                 epochs=epochs,
                                 callbacks=cbks,
                                 validation_data=(x_test, y_test))
        self.model.save('resnet.h5')

    def evaluate(self):
        (x_train, y_train), (x_test, y_test) = self.data
        self.model.load_weights('vgg19.h5', by_name=True)
        self.model.load_weights('resnet.h5', by_name=True)
        loss, acc = self.model.evaluate(x=x_test[:512, :, :, :], y=y_test[:512], batch_size=64, verbose=1)
        print('test_loss:', loss, 'test_acc:', acc)


class DenseNet(NNModel):
    def __init__(self, data, load_model=False):
        growth_rate = 12
        depth = 100
        compression = 0.5

        img_rows, img_cols = 31, 31
        img_channels = 3
        self.classes_num = 1311
        weight_decay = 0.0002
        self.data = self.set_data(data)

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
            return Dense(self.classes_num, activation='softmax', kernel_initializer=he_normal(),
                         kernel_regularizer=regularizers.l2(weight_decay))(x)

        nblocks = (depth - 4) // 6
        nchannels = growth_rate * 2

        img_input = Input(shape=(img_rows, img_cols, img_channels))
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

        self.model = Model(img_input, x)
        self.model.summary()

        if load_model:
            self.model.load_weights('densenet.h5', by_name=True)

        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def set_data(self, data):
        (x_train, y_train), (x_test, y_test) = data
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
            x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]

        y_train = keras.utils.to_categorical(y_train, self.classes_num)
        y_test = keras.utils.to_categorical(y_test, self.classes_num)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        return (x_train, y_train), (x_test, y_test)

    def train(self):
        (x_train, y_train), (x_test, y_test) = self.data
        batch_size = 64  # 64 or 32 or other
        epochs = 250
        iterations = 782

        def scheduler(epoch):
            if epoch <= 75:
                return 0.05
            if epoch <= 150:
                return 0.005
            if epoch <= 210:
                return 0.0005
            return 0.0001

        tb_cb = TensorBoard(log_dir='./densenet/', histogram_freq=0)
        change_lr = LearningRateScheduler(scheduler)
        cbks = [change_lr, tb_cb]

        # set data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125,
                                     fill_mode='constant', cval=0.)
        datagen.fit(x_train)

        # start training
        self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=iterations,
                                 epochs=epochs, callbacks=cbks, validation_data=(x_test, y_test))
        self.model.save('densenet.h5')

    def evaluate(self):
        (x_train, y_train), (x_test, y_test) = self.data
        self.model.load_weights('densenet.h5', by_name=True)
        loss, acc = self.model.evaluate(x=x_test[:512, :, :, :], y=y_test[:512], batch_size=64, verbose=1)
        print('test_loss:', loss, 'test_acc:', acc)


class ResNext(NNModel):
    def __init__(self, data, load_model=False):
        cardinality = 4  # 4 or 8 or 16 or 32
        base_width = 64
        inplanes = 64
        expansion = 4
        self.classes_num = 1311

        img_rows, img_cols = 31, 31
        img_channels = 3
        weight_decay = 0.0005

        self.data = self.set_data(data)

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
            return Dense(self.classes_num, activation='softmax', kernel_initializer=he_normal(),
                         kernel_regularizer=regularizers.l2(weight_decay))(x)

        img_input = Input(shape=(img_rows, img_cols, img_channels))
        # build the resnext model
        x = conv3x3(img_input, 64)
        x = residual_layer(x, 3, 64)
        x = residual_layer(x, 3, 128, stride=(2, 2))
        x = residual_layer(x, 3, 256, stride=(2, 2))
        x = GlobalAveragePooling2D()(x)
        x = dense_layer(x)

        self.model = Model(img_input, x)
        self.model.summary()
        if load_model:
            self.model.load_weights('resnext.h5', by_name=True)
        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def set_data(self, data):
        (x_train, y_train), (x_test, y_test) = data
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
            x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]

        y_train = keras.utils.to_categorical(y_train, self.classes_num)
        y_test = keras.utils.to_categorical(y_test, self.classes_num)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        return (x_train, y_train), (x_test, y_test)

    def train(self):
        batch_size = 120  # 64 or 32 or other
        epochs = 250
        iterations = 417

        def scheduler(epoch):
            if epoch <= 75:
                return 0.05
            if epoch <= 150:
                return 0.005
            if epoch <= 210:
                return 0.0005
            return 0.0001

        tb_cb = TensorBoard(log_dir='./resnext/', histogram_freq=0)
        change_lr = LearningRateScheduler(scheduler)
        cbks = [change_lr, tb_cb]
        (x_train, y_train), (x_test, y_test) = self.data

        # set data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125,
                                     fill_mode='constant', cval=0.)

        datagen.fit(x_train)

        # start training
        self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=iterations,
                                 epochs=epochs, callbacks=cbks, validation_data=(x_test, y_test))
        self.model.save('resnext.h5')

    def evaluate(self):
        (x_train, y_train), (x_test, y_test) = self.data
        self.model.load_weights('vgg19.h5', by_name=True)
        self.model.load_weights('resnext.h5', by_name=True)
        loss, acc = self.model.evaluate(x=x_test[:512, :, :, :], y=y_test[:512], batch_size=64, verbose=1)
        print('test_loss:', loss, 'test_acc:', acc)


class Xception(NNModel):
    def __init__(self, data, load_model=False):
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
        img_rows, img_cols = 31, 31
        img_channels = 3
        weight_decay = 0.0005
        self.classes_num = 1311
        self.data = self.set_data(data)

        img_input = Input(shape=(img_rows, img_cols, img_channels))
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

        x = Dense(self.classes_num, activation='softmax', kernel_initializer=he_normal(),
              kernel_regularizer=regularizers.l2(weight_decay))(x)

        self.model = Model(img_input, x, name='xception')
        self.model.summary()
        if not load_model:
            weights_path = get_file(
                'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
            self.model.load_weights(weights_path)
        else:
            self.model.load_weights('xception.h5')

        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def set_data(self, data):
        (x_train, y_train), (x_test, y_test) = data
        y_train = keras.utils.to_categorical(y_train, self.classes_num)
        y_test = keras.utils.to_categorical(y_test, self.classes_num)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # data preprocessing
        x_train[:, :, :, 0] = (x_train[:, :, :, 0] - 123.680)
        x_train[:, :, :, 1] = (x_train[:, :, :, 1] - 116.779)
        x_train[:, :, :, 2] = (x_train[:, :, :, 2] - 103.939)
        x_test[:, :, :, 0] = (x_test[:, :, :, 0] - 123.680)
        x_test[:, :, :, 1] = (x_test[:, :, :, 1] - 116.779)
        x_test[:, :, :, 2] = (x_test[:, :, :, 2] - 103.939)

        return (x_train, y_train), (x_test, y_test)

    def train(self):
        batch_size = 120  # 64 or 32 or other
        epochs = 250
        iterations = 417

        def scheduler(epoch):
            if epoch <= 75:
                return 0.05
            if epoch <= 150:
                return 0.005
            if epoch <= 210:
                return 0.0005
            return 0.0001

        tb_cb = TensorBoard(log_dir='./resnext/', histogram_freq=0)
        change_lr = LearningRateScheduler(scheduler)
        cbks = [change_lr, tb_cb]
        (x_train, y_train), (x_test, y_test) = self.data

        # set data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125,
                                     fill_mode='constant', cval=0.)

        datagen.fit(x_train)

        # start training
        self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=iterations,
                                 epochs=epochs, callbacks=cbks, validation_data=(x_test, y_test))
        self.model.save('xception.h5')

    def evaluate(self):
        (x_train, y_train), (x_test, y_test) = self.data
        self.model.load_weights('vgg19.h5', by_name=True)
        self.model.load_weights('resnext.h5', by_name=True)
        loss, acc = self.model.evaluate(x=x_test[:512, :, :, :], y=y_test[:512], batch_size=64, verbose=1)
        print('test_loss:', loss, 'test_acc:', acc)
