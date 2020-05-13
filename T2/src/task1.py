from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import bovw 
import os
import numpy as np


def task1_BOVW():
    bovw.training()


def task1_CNN():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    data_folder = '../res/Task 1/Training'
    test_folder = '../res/Task 1/Test'
    total_train = 900
    total_test = 379

    batch_size = 16
    epochs = 15
    labels = ["Bening", "Malignant"]

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        data_folder, class_mode='binary', batch_size=batch_size, target_size=(150, 150),)
    test_generator = test_datagen.flow_from_directory(
        test_folder, class_mode='binary', batch_size=batch_size, target_size=(150, 150),)

    # confirm the iterator works
    batchX, batchy = train_generator.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' %
          (batchX.shape, batchX.min(), batchX.max()))

    model.fit_generator(
        train_generator,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs)

    model.evaluate_generator(test_generator)

    predictions = model.predict_generator(test_generator)

    print(np.argmax(predictions[0]), labels[np.argmax(predictions[0])])

    # always save your weights after training or during training
    model.save_weights('first_try.h5')
