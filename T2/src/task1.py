from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

def task1(): 
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    # model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    # model.add(Conv2D(64, (3, 3), input_shape=(3, 150, 150)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    data_folder = '../res/ISBI2016_ISIC_Part3_Training_Data'
    test_folder = '../res/ISBI2016_ISIC_Part3_Test_Data'

    batch_size = 16

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
    validation_generator = test_datagen.flow_from_directory(
        test_folder, class_mode='binary', batch_size=batch_size, target_size=(150, 150),)

    # confirm the iterator works
    batchX, batchy = train_generator.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' %
        (batchX.shape, batchX.min(), batchX.max()))

    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
    # always save your weights after training or during training
    model.save_weights('first_try.h5')
