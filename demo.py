import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
import data_util
import neural_net

root = 'ShanghaiTech_Crowd_Counting_Dataset'
folder = 'processed'


def main():
    input_shape = (128, 170, 3)
    output_shape = (128, 170, 1)

    input_layer = layers.Input(input_shape)

    model = neural_net.init_model(input_layer, output_shape)
    model.load_weights('weights/')
    folder = 'processed'
    part_B_test = os.path.join(root, 'part_B_final', 'test_data', folder)

    X_test = data_util.load_data_without_labels(part_B_test, 316)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0012),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    test_sample = X_test[0:316]
    prediction = model.predict(test_sample)
    data_util.show_prediction_as_image(prediction[315])


if __name__ == '__main__':
    main()
