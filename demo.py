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
        image1 = image.load_img(os.path.join(demos, img), target_size=(128, 170, 3))
        image1 = image.img_to_array(image1)
        #image1 = np.expand_dims(image1, axis=0)
        images.append(image1)
    images = np.array(images)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0012),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    test_samples = images[0:4]
    prediction = model.predict(test_samples)
    data_util.show_prediction_as_image(prediction[2])


if __name__ == '__main__':
    main()
