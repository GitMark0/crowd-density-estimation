import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import  layers, models
import data_util
from tensorflow import expand_dims
from matplotlib import pyplot as plt
from PIL.Image import Image

root = 'ShanghaiTech_Crowd_Counting_Dataset'


def init_model(input_layer, output_shape):
    model = models.Sequential()
    model.add(input_layer)

    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(2, (3, 3)))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(6, (2, 2)))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(10, (3, 3)))
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(10, (3, 3)))
    model.add(layers.UpSampling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(6, (2, 2)))
    model.add(layers.UpSampling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(3, (3, 3)))
    model.add(layers.Conv2DTranspose(3, (1, 3)))

    model.summary()
    return model

def main():
    folder = 'processed'
    part_B_train = os.path.join(root, 'part_B_final', 'train_data', folder)
    part_B_test = os.path.join(root, 'part_B_final', 'test_data', folder)

    X_train = data_util.load_data_without_labels(part_B_train, 400)
    y_train = data_util.load_data_without_labels(part_B_train.replace('processed','processed_labels'), 400)
    X_test = data_util.load_data_without_labels(part_B_test, 316)
    y_test = data_util.load_data_without_labels(part_B_test.replace('processed','processed_labels'), 316)

    input_shape  = (128, 170, 3)
    output_shape = (128, 170, 1)

    test = expand_dims(X_train[0], 0)
    input_layer = layers.Input(input_shape)

    model = init_model(input_layer, output_shape)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train[0:400], epochs=5, validation_data=(X_test, y_test[0:316]), verbose=1)

    #test_sample = tf.expand_dims(X_train[0], 0)
    test_sample = X_train[0:2]
    prediction = model.predict(test_sample)
    image = data_util.image_from_arr(prediction, gray = False, norm = True)
    image.show()

    #model.save_weights('weights/')




if __name__ == '__main__':
    main()
