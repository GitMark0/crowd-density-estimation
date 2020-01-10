from keras.models import load_model
from keras_preprocessing import image
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import os
import shutil

# Setting model

model = Sequential()
input_shape = (64, 64, 3)

model.add(Conv2D(filters=512, kernel_size=(4, 4), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(filters=256, kernel_size=(2, 2), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=256, kernel_size=(2, 2), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512))

# Reduce overfitting by randomly turning neurons on and off during training
model.add(Dropout(0.4))

model.add(Dense(1))
model.add(Activation('sigmoid'))
# Load trained model
model.load_weights('top_weights_v1_crowd.h5')

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Image size
image_size = (64, 64)
# # Load image
# apple_file = 'rotten.png'
# apple_image = image.load_img(apple_file, target_size=image_size)
#
# # Turn apple_image to array
# apple_image = image.img_to_array(apple_image)
# apple_image = np.expand_dims(apple_image, axis=0)
# apple_image = apple_image / 255
#
# prediction = model.predict_classes(apple_image)[0]
# print(prediction)

# Dense - 0, Sparse - 1
rootdir = '/media/dabar/C0CA6608CA65FB54/Ostaci_projekt/VisDrone2019-DET-train/images'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = subdir + os.sep + file
        # Load image
        apple_file = filepath
        apple_image = image.load_img(apple_file, target_size=image_size)

        # Turn apple_image to array
        apple_image = image.img_to_array(apple_image)
        apple_image = np.expand_dims(apple_image, axis=0)
        apple_image = apple_image / 255

        prediction = model.predict_classes(apple_image)[0]
        print(prediction, file)
        # if prediction != 0:
        #     shutil.copy(filepath, '/home/dabar/Desktop/failed_apples/')
        #     print(filepath)
