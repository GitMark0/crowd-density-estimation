from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense
from keras_preprocessing.image import ImageDataGenerator, warnings
import tensorflow as tf
from keras.callbacks import Callback, ModelCheckpoint


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='accuracy', value=0.97, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current <= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# K.tensorflow_backend._get_available_gpus()

callbacks = [
    EarlyStoppingByLossVal(monitor='loss', value=0.01, verbose=1),
    # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ModelCheckpoint('top_weights_v2_crowd.h5', monitor='accuracy', save_best_only=True, verbose=0),
]

input_size = (128, 128, 3)
model = Sequential()

## FIRST SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=128, kernel_size=(2, 2), input_shape=input_size, activation='relu', ))
# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=128, kernel_size=(2, 2), input_shape=input_size, activation='relu', ))

# POOLING LAYER
model.add(MaxPooling2D(pool_size=(8, 8)))

## SECOND SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=256, kernel_size=(2, 2), input_shape=input_size, activation='relu', ))
# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=256, kernel_size=(2, 2), input_shape=input_size, activation='relu', ))

# POOLING LAYER
model.add(MaxPooling2D(pool_size=(8, 8)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 512 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 4

image_gen = ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=1 / 255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')

train_image_gen = image_gen.flow_from_directory(
    '/media/dabar/C0CA6608CA65FB54/PycharmProjects/Raspoznavanje_gustoca/Dataset/train',
    target_size=input_size[:2],
    batch_size=batch_size,
    class_mode='binary')

test_image_gen = image_gen.flow_from_directory(
    '/media/dabar/C0CA6608CA65FB54/PycharmProjects/Raspoznavanje_gustoca/Dataset/test',
    target_size=input_size[:2],
    batch_size=batch_size,
    class_mode='binary')

print(train_image_gen.class_indices)

results = model.fit_generator(train_image_gen, callbacks=callbacks,
                              validation_data=test_image_gen, validation_steps=12, epochs=10000)

model.save_weights('top_weights_v2_crowd.h5')
