from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import os
import numpy as np
import csv
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from sklearn.utils import class_weight


labels = ["Actinic keratosis", "Basal cell carcinoma", "Benign keratosis", "Dermatofibroma", "Melanocytic nevus", "Melanoma", "Vascular lesion"]

def task2():
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
    model.add(Dense(len(labels)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    data_folder = '../res/Task 2/Training'
    test_folder = '../res/Task 2/Test'
    total_train = 7010
    total_test = 3005

    batch_size = 16
    epochs = 1

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
        data_folder, class_mode='categorical', batch_size=batch_size, target_size=(150, 150),)
    test_generator = test_datagen.flow_from_directory(
        test_folder, class_mode='categorical', batch_size=batch_size, target_size=(150, 150),)

    counter = Counter(train_generator.classes)                          
    max_val = float(max(counter.values()))       
    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}              

    # confirm the iterator works
    batchX, batchy = train_generator.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' %
          (batchX.shape, batchX.min(), batchX.max()))

    model.fit_generator(
        train_generator,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        class_weight=class_weights)

    scores = model.evaluate_generator(test_generator, total_test // batch_size)

    print("Test accuracy = ", scores[1])

    predictions = model.predict_generator(test_generator, total_test // batch_size + 1)
    y_pred = np.argmax(predictions, axis=1)
    print(np.argmax(predictions[0]), labels[np.argmax(predictions[0])])

    with open('results2.csv', mode="w") as results_file:
        writer = csv.writer(results_file, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for x in predictions:
            writer.writerow(x)

    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    print('Classification Report')
    print(classification_report(test_generator.classes, y_pred, target_names=labels))

    # always save your weights after training or during training
    model.save_weights('first_try.h5')
