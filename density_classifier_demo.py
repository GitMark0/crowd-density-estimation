import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import data_util
import os
import density_classifier_neural_net as neural_net
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

input_size = (64, 128, 3)

def test_dataset_examples(model):
    root = 'beijing'
    batch_size = 4
    image_gen = ImageDataGenerator(rescale=1 / 255,
                               fill_mode='nearest')

    test_image_gen = image_gen.flow_from_directory(
        os.path.join(root, 'test'),
        target_size=input_size[:2],
        batch_size=batch_size,
        class_mode='categorical')

    loss,acc = model.evaluate_generator(generator=test_image_gen)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))





def main():

    model = neural_net.init_model()
    model.load_weights('weights/density_classifier/cp.ckpt')
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    test_dataset_examples(model)


if __name__ == '__main__':
    main()
