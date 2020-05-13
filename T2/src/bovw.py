from scipy.cluster.vq import kmeans, vq
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score  # sreeni
from scipy.cluster.vq import vq
import cv2
import numpy as np
import os


# To make it easy to list all file names in a directory let us define a function
def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


def getInformation(path):

    names = os.listdir(path)

    # Get path to all images and save them in a list
    # image_paths and the corresponding label in image_paths
    image_paths = []
    image_classes = []
    class_id = 0

    # Fill the placeholder empty lists with image path, classes, and add class ID number
    for name in names:
        dir = os.path.join(path, name)
        class_path = imglist(dir)
        image_paths += class_path
        image_classes += [class_id]*len(class_path)
        class_id += 1

    return image_paths, image_classes, names


def featureExtraction(image_paths, isTraining):

    # Create feature extraction and keypoint detector objects
    # SIFT is not available anymore in openCV
    # Create List where all the descriptors will be stored
    des_list = []

    # BRISK is a good replacement to SIFT. ORB also works but didn;t work well for this example
    brisk = cv2.BRISK_create(30)
    print(brisk.descriptorSize())
    for image_path in image_paths:
        im = cv2.imread(image_path)
        _, des = brisk.detectAndCompute(im, None)
        des_list.append((image_path, des))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for _, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    if isTraining is True:
        # kmeans works only on float, so convert integers to float
        descriptors_float = descriptors.astype(float)
        return des_list, descriptors_float
    else:
        return des_list, descriptors


def KMeansCluster(descriptors, des_list, image_paths):
    # Perform k-means clustering and vector quantization
    k = 200  # k means with 100 clusters gives lower accuracy for the aeroplane example
    voc, _ = kmeans(descriptors, k, 1)

    # Calculate the histogram of features and represent them as vector
    # vq Assigns codes from a code book to observations.
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in range(len(image_paths)):
        words, _ = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    return im_features, k, voc


# def tfIdf(im_features, image_paths):
# 	# Perform Tf-Idf vectorization
# 	nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
# 	idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')


def normalization(im_features):
    # Scaling the words
    # Standardize features by removing the mean and scaling to unit variance
    # In a way normalization
    stdSlr = StandardScaler().fit(im_features)
    new_im_features = stdSlr.transform(im_features)

    return stdSlr, new_im_features


def storeSVM(clf, training_names, stdSlr, k, voc):
    # Save the SVM
    # Joblib dumps Python object into one file
    joblib.dump((clf, training_names, stdSlr, k, voc), "bovw.pkl", compress=3)


def training():

    image_paths, image_classes, training_names = getInformation(
        '../res/ISBI2016_ISIC_Part3_Training_Data')
    des_list, descriptors = featureExtraction(image_paths, True)
    im_features, k, voc = KMeansCluster(descriptors, des_list, image_paths)
    stdSlr, im_features = normalization(im_features)

    # Train an algorithm to discriminate vectors corresponding to positive and negative training images
    # Train the Linear SVM
    # Default of 100 is not converging
    clf = LinearSVC(max_iter=10000)
    clf.fit(im_features, np.array(image_classes))

    # Train Random forest to compare how it does against SVM
    # from sklearn.ensemble import RandomForestClassifier
    #clf = RandomForestClassifier(n_estimators = 100, random_state=30)
    #clf.fit(im_features, np.array(image_classes))

    storeSVM(clf, training_names, stdSlr, k, voc)


def calcFeatureHistogram(image_paths, des_list, stdSlr, k, voc):

    # Calculate the histogram of features
    # vq Assigns codes from a code book to observations.
    test_features = np.zeros((len(image_paths), k), "float32")

    for i in range(len(image_paths)):
        words, _ = vq(des_list[i][1], voc)
    for w in words:
        test_features[i][w] += 1

    # Perform Tf-Idf vectorization
    # nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
    # idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

        # Scale the features
        # Standardize features by removing the mean and scaling to unit variance
        # Scaler (stdSlr comes from the pickled file we imported)
    test_features = stdSlr.transform(test_features)

    return test_features


def confusionMatrix(true_class, predictions):
    accuracy = accuracy_score(true_class, predictions)
    print("accuracy = ", accuracy)
    cm = confusion_matrix(true_class, predictions)
    print(cm)


def validate():

    clf, classes_names, stdSlr, k, voc = joblib.load("bovw.pkl")

    image_paths, image_classes, _ = getInformation(
        '..res/ISBI2016_ISIC_Part3_Test_Data')
    des_list, _ = featureExtraction(image_paths, False)

    test_features = calcFeatureHistogram(image_paths, des_list, stdSlr, k, voc)

    # Report true class names so they can be compared with predicted classes
    true_class = [classes_names[i] for i in image_classes]
    # Perform the predictions and report predicted class names.
    predictions = [classes_names[i] for i in clf.predict(test_features)]

    # Print the true class and Predictions
    print("true_class =" + str(true_class))
    print("prediction =" + str(predictions))

    confusionMatrix(true_class, predictions)
