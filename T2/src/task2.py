from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.applications import MobileNet
import os
import numpy as np
import csv
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from sklearn.utils import class_weight
from utils import plot_cm


def task2():
    # Create a MobileNet model
    mobile = MobileNet(weights='imagenet')

    # See a summary of the layers in the model
    mobile.summary()

    # Modify the model
    # Exclude the last 5 layers of the model
    x = mobile.layers[-6].output
    # Add a dropout and dense layer for predictions
    x = Dropout(0.25)(x)
    predictions = Dense(7, activation='softmax')(x)

    # Create a new model with the new outputs
    model = Model(inputs=mobile.input, outputs=predictions)

    # See a summary of the new layers in the model
    model.summary()

    # Freeze the weights of the layers that we aren't training (training the last 23)
    for layer in model.layers[:-23]:
        layer.trainable = False


    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Useful variables
    data_folder = '../res/Task 2/Training'
    test_folder = '../res/Task 2/Test'
    total_train = 8012
    total_test = 2003
    labels = ["AK", "BCC", "BK", "D", "MN", "M", "VL"]
    batch_size = 100 
    epochs = 10

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
        data_folder, class_mode='categorical', batch_size=batch_size, target_size=(224, 224),)
    test_generator = test_datagen.flow_from_directory(
        test_folder, class_mode='categorical', batch_size=batch_size, target_size=(224, 224))
    
    # Try to deal with class imbalance: calculate class_weights so that the minority classes have a larger weight
    # than the majority classes.
    class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_generator.classes), 
                train_generator.classes)
    class_weights = dict(enumerate(class_weights))

    # Train the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        class_weight=class_weights
        )

    # Evaluate the model accuracy with the testing dataset
    scores = model.evaluate_generator(test_generator, total_test // batch_size)
    print("Test accuracy = ", scores[1])

    # Generate predictions with the test dataset
    # softmax returns a value for each class
    # the predicted class for a given sample will be the one that has the maximum value
    predictions = model.predict_generator(test_generator, total_test // batch_size + 1)
    y_pred = np.argmax(predictions, axis=1)

    # Save the predictions in a csv file
    with open('results2.csv', mode="w") as results_file:
        writer = csv.writer(results_file, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for x in predictions:
            writer.writerow(x)

    # Generate confusion matrix and classification report
    # Helps to evaluate metrics such as accuracy, precision, recall
    print('Confusion Matrix')
    cm = confusion_matrix(test_generator.classes, y_pred)
    print(cm)
    plot_cm(cm, labels, "second.png")

    print('Classification Report')
    print(classification_report(test_generator.classes, y_pred, target_names=labels))
