import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import data_util
import os
import neural_net
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

input_size = (64, 128, 3)

def test_dataset_examples(model):
    root = 'beijing2'
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


def test_random_examples(model, example_index):

    root = 'test_examples'
    examples = ['82063735_2863425327047082_1096112625017683968_n.jpg',
        '82802453_642298606511069_9138862684085682176_n.jpg',
        '82428967_753940481794855_6610497251762503680_n.jpg', 'IMG_0.jpg', 'IMG_1.jpg',
        'IMG_2.jpg','IMG_3.jpg','IMG_4.jpg', 'IMG_5.jpg']

    _, X = data_util.load_example(os.path.join(root, examples[example_index]), input_size)
    print('Prediction: ', model.predict(np.expand_dims(X, axis=0)))


def main():

    model = neural_net.init_model()
    model.load_weights('weights/cp.ckpt')
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    #test_dataset_examples(model)
    test_random_examples(model, 8)


if __name__ == '__main__':
    main()
