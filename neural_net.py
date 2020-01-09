import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import data_util
import os

root = 'ShanghaiTech_Crowd_Counting_Dataset'


def init_model():

    model = models.Sequential()
    model.add(layers.Conv2D(3, (3, 3), activation='relu', input_shape=(191, 255, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(6, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(9, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()
    return model



def main():
    part_B_train = os.path.join(root,'part_B_final','train_data','images')
    part_B_test = os.path.join(root,'part_B_final','test_data','images')
    path_sets = [(part_B_train, 1), (part_B_test, 1)]

    X_train, y_train = data_util.load_data(path_sets[0][0], path_sets[0][1])
    X_test, y_test = data_util.load_data(path_sets[1][0], path_sets[1][1])

    model = init_model()

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    history = model.fit(X_train, y_train[0:path_sets[0][1]], epochs=10,
                    validation_data=(X_test, y_test[0:path_sets[1][1]]))

    model.save_weights('weights/')


if __name__ == '__main__':
    main()
