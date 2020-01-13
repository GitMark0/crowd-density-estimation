"""
Crowd density estimation
This is a final implementation version.

The program uses two built models for crowd density estimation. The models were trained on datasets mentioned in
documentation. Models give their different estimation for the same test picture:
1) First model calculates crowd density on the image and classifies it in one of three categories:
low, average and high density
2) Second model predicts a 2D heat map showing the intensity of crowd in every image area

There are also two separate demo files (density_classifier_demo.py, heat_map_demo.py) demonstrating separate predictions
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
import data_util
import heat_map_neural_net
import density_classifier_neural_net
import numpy as np
from matplotlib import pyplot as plt
import sys


class_interpretations = {0: 'Low density', 1: 'Medium density', 2: 'High density'}
heat_map_input_shape = (128, 170, 3)
density_input_shape = (64, 128, 3)


def get_heatmap_model():
    input_layer = layers.Input(heat_map_input_shape)
    heat_map_weights = 'weights/heat_map/'
    heat_map_model = heat_map_neural_net.init_model(input_layer)
    heat_map_model.load_weights(heat_map_weights)
    heat_map_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0012),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=['accuracy'],
                           sample_weight_mode='temporal')
    return heat_map_model

def get_classifier_model():

    classifier_weights = 'weights/density_classifier/cp.ckpt'
    classifier_model = density_classifier_neural_net.init_model()
    classifier_model.load_weights(classifier_weights)
    classifier_model.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
    return classifier_model

if __name__ == '__main__':

    arg = sys.argv[1]
    test_image = 'test_examples/IMG_' + str(arg) + '.jpg'

    heat_map_model = get_heatmap_model()
    classifier_model = get_classifier_model()

    #  load images for heat map generator
    (size, rotate), heat_image = \
            data_util.load_example(test_image, heat_map_input_shape)

    # load images for density classifier
    _, density_image = data_util.load_example(test_image, density_input_shape)

    #generate and plot heat map
    heat_map = heat_map_model.predict(np.expand_dims(heat_image, axis=0))[0]
    data_util.show_prediction_as_image(heat_map, size, rotate)

    #predict density class
    probability = np.round(classifier_model.predict(np.expand_dims(density_image, axis=0))[0], 3)*100
    classification = np.argmax(probability)
    interpretation = class_interpretations[classification]

    plt.title('Probabilistic output: {}\n Class Index: {}\n Interpretation: {}'.format(probability, classification, interpretation))

    plt.show()
