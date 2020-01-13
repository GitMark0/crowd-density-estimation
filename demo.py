import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import data_util
import neural_net
import numpy as np
import matplotlib.pyplot as plt


def main():
    input_shape = (128, 170, 3)
    output_shape = (128, 170, 3)

    input_layer = layers.Input(input_shape)

    model = neural_net.init_model(input_layer, output_shape)
    model.load_weights('weights/')
    demos = 'demo_pictures'

    images = []
    for img in ['IMG_1.jpg', 'IMG_2.jpg', 'IMG_3.jpg', 'IMG_4.jpg', 'IMG_5.jpg']:
        _, image1 = data_util.load_example(os.path.join(demos, img), input_shape)
        images.append(image1)
    images = np.array(images)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0012),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    test_samples = images[0:4]
    predictions = model.predict(test_samples)
    for prediction in predictions:
        data_util.show_prediction_as_image(prediction)


if __name__ == '__main__':
    main()
