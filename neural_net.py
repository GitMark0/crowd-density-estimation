import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import data_util
import os

root = 'ShanghaiTech_Crowd_Counting_Dataset'


def init_model():

    model = models.Sequential()
    model.add(layers.Conv2D(3, (5, 5), activation='relu', input_shape=(128, 170, 3)))
    model.add(layers.Conv2D(9, (4, 4), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(27, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(54, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(100, (2, 2), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.summary()
    return model



def main():
    folder = 'processed'
    part_B_train = os.path.join(root,'part_B_final','train_data',folder)
    part_B_test = os.path.join(root,'part_B_final','test_data',folder)
    path_sets = [(part_B_train, 400), (part_B_test, 316)]

    X_train, y_train = data_util.load_data(path_sets[0][0], path_sets[0][1])
    X_test, y_test = data_util.load_data(path_sets[1][0], path_sets[1][1])


    model = init_model()

    model.compile(optimizer='adadelta',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    history = model.fit(X_train, y_train[0:path_sets[0][1]], epochs=50,
                    validation_data=(X_test, y_test[0:path_sets[1][1]]))



    model.save_weights('weights/')


if __name__ == '__main__':
    main()
