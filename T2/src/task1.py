from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import bovw 
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from collections import Counter

import csv


def task1_BOVW():
    bovw.training()
    bovw.validate()


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
    labels = ["bening", "malignant"]

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        data_folder, class_mode='binary', batch_size=batch_size, target_size=(150, 150),)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_folder, class_mode='binary', batch_size=batch_size, target_size=(150, 150),)

    counter = Counter(train_generator.classes)                          
    max_val = float(max(counter.values()))       
    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}                     


    model.fit_generator(
        train_generator,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        class_weight=class_weights)

    scores = model.evaluate_generator(test_generator, total_test // batch_size)

    print("Test accuracy = ", scores[1])

    predictions = model.predict_generator(
        test_generator, total_test // batch_size + 1)
    predicted_classes = [1 * (x[0]>=0.5) for x in predictions]
    print(np.argmax(predicted_classes[0]),
          labels[np.argmax(predicted_classes[0])])

    with open('results.csv', mode="w") as results_file:
        writer = csv.writer(results_file, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for x in predictions:
            writer.writerow(x)

    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    print('Confusion Matrix')
    print(confusion_matrix(true_classes, predicted_classes))
    print('Classification Report')
    print(classification_report(true_classes,
                                predicted_classes, target_names=class_labels))

    # always save your weights after training or during training
    model.save_weights('first_try.h5')
