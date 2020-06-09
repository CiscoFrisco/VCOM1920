from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
import bovw 
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from collections import Counter
from utils import plot_cm

import csv


def task1_BOVW():
    bovw.training()
    bovw.validate()


def task1_CNN():
    # Build the model
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
   
    # this converts our 3D feature maps to 1D feature vectors
    # add 1 dropout layer in order to prevent overfitting
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # Useful variables
    data_folder = '../res/Task 1/Training'
    test_folder = '../res/Task 1/Test'
    total_train = 900
    total_test = 379
    batch_size = 100 # Higher batch size than usual in order to have a higher probability of encountering malignant samples in each batch
    epochs = 50
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

    # Try to deal with class imbalance: calculate class_weights so that the malignant class has a larger weight
    # than the bening class.
    counter = Counter(train_generator.classes)                          
    max_val = float(max(counter.values()))       
    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}                     

    # Train the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        class_weight=class_weights)

    # Evaluate the model accuracy with the testing dataset
    scores = model.evaluate_generator(test_generator, total_test // batch_size)
    print("Test accuracy = ", scores[1])

    # Generate predictions with the test dataset
    # sigmoid returns a value between 0 and 1, with 0.5
    # if the value is lower than 0.5, then the model believes the sample is bening
    # if the value is bigger than 0.5, then the model believes the sample is malignant
    # The lower the value (close to 0), the most confidence the sample belongs to the bening class
    # The higher the value (close to 1), the most confidence the sample belongs to the malignant class
    predictions = model.predict_generator(
        test_generator, total_test // batch_size + 1)
    predicted_classes = [1 * (x[0]>=0.5) for x in predictions]

    # Save the predictions in a csv file
    with open('results.csv', mode="w") as results_file:
        writer = csv.writer(results_file, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for x in predictions:
            writer.writerow(x)

    # Generate confusion matrix and classification report
    # Helps to evaluate metrics such as accuracy, precision, recall
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    print('Confusion Matrix')
    cm = confusion_matrix(true_classes, predicted_classes)
    print(cm)
    plot_cm(cm, labels, "first.png")
    
    print('Classification Report')
    print(classification_report(true_classes,
                                predicted_classes, target_names=class_labels))
