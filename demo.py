import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import data_util
import os
import neural_net
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

root = 'test_examples'
examples = ['82063735_2863425327047082_1096112625017683968_n.jpg',
    '82802453_642298606511069_9138862684085682176_n.jpg',
    '82428967_753940481794855_6610497251762503680_n.jpg']

def main():

    input_size = (64, 128, 3)

    _, X = data_util.load_example(os.path.join(root, examples[2]), input_size)

    model = neural_net.init_model()
    model.load_weights('weights/')
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    print('Prediction: ', model.predict(np.expand_dims(X, axis=0)))



if __name__ == '__main__':
    main()
