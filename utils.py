from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from options import *


def get_scheduler(knots):
    def scheduler(epoch):
        for k in knots:
            if epoch <= k[0]:
                return k[1]
        return knots[-1][1]

    return scheduler


def update_config(cfg):
    selected_option = OPTIONS[cfg['model']]
    for k in selected_option:
        cfg[k] = selected_option[k]
    return cfg


def set_data(data, cfg):
    num_classes = cfg['num_classes']
    (x_train, y_train), (x_test, y_test) = data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    mean = cfg['mean']
    std = cfg['std']
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return (x_train, y_train), (x_test, y_test)


def train(data, model, cfg):
    (x_train, y_train), (x_test, y_test) = data
    batch_size = cfg['batch_size']
    epochs = cfg['epochs']
    iterations = cfg['iterations']
    log_filepath = cfg['log_filepath']
    scheduler = get_scheduler(cfg['scheduler'])

    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb]

    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant', cval=0.)

    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        callbacks=cbks,
                        validation_data=(x_test, y_test))
    model.save(cfg['name'] + '.h5')


def evaluate(data, model, num=-1):
    (x_train, y_train), (x_test, y_test) = data
    loss, acc = model.evaluate(x=x_test[:num, :, :, :], y=y_test[:num], batch_size=64, verbose=1)
    print('test_loss:', loss, 'test_acc:', acc)
