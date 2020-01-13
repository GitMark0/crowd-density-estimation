import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import data_util
import os
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

root = 'beijing2'


def init_model():

    model = models.Sequential()
    model.add(layers.Conv2D(3, (5, 5), activation='relu', input_shape=(64, 128, 3)))
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

    return model



def main():

    input_size = (64, 128, 3)
    batch_size = 4
    image_gen = ImageDataGenerator(rescale=1 / 255,
                               fill_mode='nearest')

    train_image_gen = image_gen.flow_from_directory(
    os.path.join(root, 'train'),
    target_size=input_size[:2],
    batch_size=batch_size,
    class_mode='categorical')

    test_image_gen = image_gen.flow_from_directory(
        os.path.join(root, 'test'),
        target_size=input_size[:2],
        batch_size=batch_size,
        class_mode='categorical')

    print(train_image_gen.class_indices)

    model = init_model()
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    callbacks = [ModelCheckpoint('weights/cp.ckpt', monitor='val_acc', save_best_only=True)]

    results = model.fit_generator(train_image_gen, validation_data=test_image_gen,
        callbacks=callbacks, verbose=1, epochs=15)

    #model.save_weights('weights/')


if __name__ == '__main__':
    main()
