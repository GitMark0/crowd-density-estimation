import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import data_util
import os
import neural_net

root = 'ShanghaiTech_Crowd_Counting_Dataset'
folder = 'processed'

def main():

    model = neural_net.init_model()
    model.load_weights('weights/')
    part_B_test = os.path.join(root,'part_B_final','test_data',folder)
    X_test, y_test = data_util.load_data(part_B_test, 10)

    loss,acc = model.evaluate(X_test,  y_test[0:10], verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))


if __name__ == '__main__':
    main()
