import numpy as np


def data_load_fir():
    f = open('val_data.bin', 'rb')
    x_test = []
    y_test = []
    for i in range(20976):
        label = f.read(2)
        label = int(label[0]) * 100 + int(label[1])
        image_r = f.read(31 * 31)
        image_g = f.read(31 * 31)
        image_b = f.read(31 * 31)
        image_r_c = []
        image_g_c = []
        image_b_c = []
        for j in range(31*31):
            image_r_c.append(int(image_r[j]))
            image_g_c.append(int(image_g[j]))
            image_b_c.append(int(image_b[j]))
        x_test.append([image_r_c, image_g_c, image_b_c])
        y_test.append(label)
    x_test = np.array(x_test).transpose([0,2,1]).reshape([-1,31,31,3])
    y_test = np.array(y_test).reshape([-1,1])

    f = open('train_data.bin', 'rb')
    x_train = []
    y_train = []
    for i in range(41952):
        label = f.read(2)
        label = int(label[0]) * 100 + int(label[1])
        image_r = f.read(31 * 31)
        image_g = f.read(31 * 31)
        image_b = f.read(31 * 31)
        image_r_c = []
        image_g_c = []
        image_b_c = []
        for j in range(31*31):
            image_r_c.append(int(image_r[j]))
            image_g_c.append(int(image_g[j]))
            image_b_c.append(int(image_b[j]))
        x_train.append([image_r_c, image_g_c, image_b_c])
        y_train.append(label)
    x_train = np.array(x_train).transpose([0,2,1]).reshape([-1,31,31,3])
    y_train = np.array(y_train).reshape([-1,1])

    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)

    return (x_train, y_train), (x_test, y_test)


def data_load():
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')

    return (x_train, y_train), (x_test, y_test)
