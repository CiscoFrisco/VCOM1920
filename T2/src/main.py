from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

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

train_it = train_datagen.flow_from_directory(
    data_folder, class_mode='binary', batch_size=batch_size, target_size=(150, 150),)
test_it = test_datagen.flow_from_directory(
    test_folder, class_mode='binary', batch_size=batch_size, target_size=(150, 150),)

# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' %
      (batchX.shape, batchX.min(), batchX.max()))
