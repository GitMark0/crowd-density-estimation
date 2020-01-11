import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import data_util
import os
import neural_net
from keras_preprocessing.image import ImageDataGenerator


root = 'beijing'
folder = 'processed'

def main():

    input_size = (64, 128, 3)
    batch_size = 4
    image_gen = ImageDataGenerator(rescale=1 / 255,
                               fill_mode='nearest')

    test_image_gen = image_gen.flow_from_directory(
        os.path.join(root, 'test'),
        target_size=input_size[:2],
        batch_size=batch_size,
        class_mode='categorical')

    model = neural_net.init_model()
    model.load_weights('weights/')
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



    loss,acc = model.evaluate_generator(generator=test_image_gen)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))


if __name__ == '__main__':
    main()
