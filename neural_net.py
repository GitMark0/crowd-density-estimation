import tensorflow as tf
from numpy.conftest import pytest_addoption
from tensorflow.keras import datasets, layers, models
import data_util
import os
from tensorflow import random_normal_initializer

root = 'ShanghaiTech_Crowd_Counting_Dataset'


def init_models():
    """
    model = models.Sequential()
    model.add(layers.Conv2D(3, (3, 3), activation='relu', input_shape=(25, 34, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(6, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(9, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()
    return model
    """

def downsample(filt, size, apply_batchnorm=True):
    model1 = models.Sequential()
    model1.add(layers.Conv2D(filt, size, strides=2, padding='same', use_bias=False, kernel_initializer=random_normal_initializer(0., 0.02)))
    if apply_batchnorm:
        model1.add(layers.BatchNormalization())
    model1.add(layers.LeakyReLU())
    return model1

def upsample(filt, size, apply_dropout=False):
    model2 = models.Sequential()
    model2.add(layers.Conv2D(filt, size, strides=2, padding='same', use_bias=False, kernel_initializer=random_normal_initializer(0., 0.02)))
    model2.add(layers.BatchNormalization())
    if apply_dropout:
        model2.add(layers.Dropout(0.5))
    model2.add(layers.ReLU())
    return model2


def main():
    folder = 'processed'
    part_B_train = os.path.join(root, 'part_B_final', 'train_data', folder)
    part_B_test = os.path.join(root, 'part_B_final', 'test_data', folder)
    path_sets = [(part_B_train, 1), (part_B_test, 1)]

    X_train, y_train = data_util.load_data(path_sets[0][0], path_sets[0][1])
    X_test, y_test = data_util.load_data(path_sets[1][0], path_sets[1][1])

    inputs = layers.Input(shape=[25, 34, 3])
    """
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]
    """
    last = layers.Conv2DTranspose(3, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=random_normal_initializer(0., 0.02),
                                  activation='tanh')

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)

    """
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # history = model.fit(X_train, y_train[0:path_sets[0][1]], epochs=10,
    #                validation_data=(X_test, y_test[0:path_sets[1][1]]))
    model.fit(X_train, y_train[0:path_sets[0][1]], epochs=10)

    model.save_weights('weights/')
    """



if __name__ == '__main__':
    main()
