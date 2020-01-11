import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
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
    part_B_train = os.path.join(root, 'part_B_final', 'train_data', folder)
    part_B_test = os.path.join(root, 'part_B_final', 'test_data', folder)

    X_train = data_util.load_data_without_labels(part_B_train, 400)
    y_train = data_util.load_data_without_labels(part_B_train.replace('processed', 'processed_labels'), 400)
    X_test = data_util.load_data_without_labels(part_B_test, 316)
    y_test = data_util.load_data_without_labels(part_B_test.replace('processed', 'processed_labels'), 316)

    model.compile(optimizer='adadelta',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    #loss,acc = model.evaluate(X_test,  y_test[0:316], verbose=2)
    #print("Restored model, accuracy: {:5.2f}%".format(100*acc))

    test_sample = X_test[0:10]
    prediction = model.predict(test_sample)
    #print(prediction[1])
    image = data_util.image_from_arr(prediction[5], gray = False, norm = True)
    image.show()

if __name__ == '__main__':
    main()
