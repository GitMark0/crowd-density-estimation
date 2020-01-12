import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import data_util
import neural_net
import numpy as np

root = 'beijing'
examples = [('1', '2-20170506192614.jpg'), ('1', '2-20170506194615.jpg'),
    ('2', '1-20170419095422.jpg'), ('2', '1-20170506103732.jpg'),
    ('3', '1-20170419094030.jpg'), ('3', '1-20170419094437.jpg')]
def main():

    example = examples[2]

    input_shape = (128, 170, 3)
    output_shape = (128, 170, 1)

    input_layer = layers.Input(input_shape)

    model = neural_net.init_model(input_layer, output_shape)
    model.load_weights('weights/')
    folder = 'processed'
    path = os.path.join(root, 'test')


    model.compile(optimizer='adadelta',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    #loss,acc = model.evaluate(X_test,  y_test[0:316], verbose=2)
    #print("Restored model, accuracy: {:5.2f}%".format(100*acc))

    (original_dim, rotated), X = data_util.load_example(os.path.join(path, example[0], example[1]), input_shape)
    prediction = model.predict(np.expand_dims(X, axis=0))
    #print(prediction[1])
    image = data_util.image_from_arr(prediction[0], gray = False, norm = True)
    if rotated:
        image.rotate(270)
    image.resize(original_dim).show()


if __name__ == '__main__':
    main()
