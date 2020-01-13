import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
import data_util
import heat_map_neural_net
import numpy as np


def main():
    input_shape = (128, 170, 3)
    input_layer = layers.Input(input_shape)

    model = heat_map_neural_net.init_model(input_layer)
    model.load_weights('weights/heat_map/')

    demos = 'test_examples'
    image_names = os.listdir(demos)

    images = []
    rotations = []
    for img in image_names:
        (size, rotate), image = data_util.load_example(os.path.join(demos, img), input_shape)
        rotations.append((size, rotate))
        images.append(image)
    images = np.array(images)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0012),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    test_samples = images[:]
    predictions = model.predict(test_samples)
    for prediction, (size, rotate) in zip(predictions, rotations):
        data_util.show_prediction_as_image(prediction, size, rotate)


if __name__ == '__main__':
    main()
