from scipy.cluster.vq import kmeans, vq, whiten
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score  # sreeni
from scipy.cluster.vq import vq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import numpy as np
import os
import time


# To make it easy to list all file names in a directory let us define a function
def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


def getInformation(path):

    names = os.listdir(path)

    # Get path to all images and save them in a list
    # image_paths and the corresponding label in image_classes
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
    brisk = cv2.BRISK_create(thresh=10, octaves=1)
    for image_path in image_paths:
        im = cv2.imread(image_path)
        small_im = cv2.resize(im, (178, 218), interpolation=cv2.INTER_AREA)
        _, des = brisk.detectAndCompute(small_im, None)
        des_list.append((image_path, des))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
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

    return k, voc


def calcFeatureHistogram(image_paths, des_list, k, voc):

    # Calculate the histogram of features
    # vq Assigns codes from a code book to observations.
    features = np.zeros((len(image_paths), k), "float32")

    for i in range(len(image_paths)):
        words, _ = vq(des_list[i][1], voc)
        for w in words:
            features[i][w] += 1

    return features


def normalizationTraining(im_features):
    # Scaling the words
    # Standardize features by removing the mean and scaling to unit variance
    # In a way normalization
    stdSlr = StandardScaler().fit(im_features)
    new_im_features = stdSlr.transform(im_features)

    return stdSlr, new_im_features


def normalizationTest(features, stdSlr):
    # Scale the features
    # Standardize features by removing the mean and scaling to unit variance
    # Scaler (stdSlr comes from the pickled file we imported)
    features = stdSlr.transform(features)
    return features


def training():

    start_time = time.time()

    image_paths, image_classes, training_names = getInformation(
        '../res/Task 1/Training')
    print('Training Information Loaded...')

    des_list, descriptors = featureExtraction(image_paths, True)
    print("Feature Extraction Finished...")

    k, voc = KMeansCluster(descriptors, des_list, image_paths)
    print("Clustering Finished...")

    im_features = calcFeatureHistogram(image_paths, des_list, k, voc)
    print("Feature Histogram calculated...")

    stdSlr, im_features = normalizationTraining(im_features)
    print("Normalization Finished...")

    # Train an algorithm to discriminate vectors corresponding to positive and negative training images
    # Train the Linear SVM - Default of 100 is not converging
    clf = LinearSVC(max_iter=1000000)
    clf.fit(im_features, np.array(image_classes))
    print("Training finished...")

    # Train Random forest to compare how it does against SVM
    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier(n_estimators=100, random_state=30)
    # clf.fit(im_features, np.array(image_classes))

    joblib.dump((clf, training_names, stdSlr, k, voc), "bovw.pkl", compress=3)
    print("Model stored!")
    print("Training Time: %s seconds" % (time.time() - start_time))


def validate():

    # load the classifier, class names, scaler, number of clusters and vocabulary
    clf, classes_names, stdSlr, k, voc = joblib.load("bovw.pkl")
    print("Model loaded...")

    image_paths, image_classes, _ = getInformation(
        '../res/Task 1/Test')
    print('Test Information Loaded...')

    des_list, _ = featureExtraction(image_paths, False)
    print("Feature Extraction Finished...")

    test_features = calcFeatureHistogram(image_paths, des_list, k, voc)
    print("Feature Histogram calculated...")

    test_features = normalizationTest(test_features, stdSlr)
    print("Normalization Finished...")

    # Report true class names so they can be compared with predicted classes
    true_class = [classes_names[i] for i in image_classes]
    # Perform the predictions and report predicted class names.
    predictions = [classes_names[i] for i in clf.predict(test_features)]

    accuracy = accuracy_score(true_class, predictions)
    cm = confusion_matrix(true_class, predictions)

    print(cm)
    print(accuracy)

    # Print the Results (True Class and Prediction, Accuracy and Confusion Matrix)
    printResults(true_class, predictions, accuracy, cm)


def printResults(true_class, predictions, accuracy, cm):

    accuracy_benign = float(cm[0][0])/(cm[0][0]+cm[0][1])
    accuracy_malignant = float(cm[1][1])/(cm[1][0]+cm[1][1])

    results = make_subplots(
        rows=2, cols=2,
        shared_xaxes=True,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.03,
        specs=[[{"colspan": 2, "type": "table"}, None],
               [{"colspan": 1, "type": "table"}, {"colspan": 1, "type": "table"}]]
    )

    results.add_trace(go.Table(header=dict(values=['True Class', 'Prediction']),
                               cells=dict(values=[true_class, predictions])
                               ), row=1, col=1
                      )

    results.add_trace(
        go.Table(
            header=dict(
                values=["", "Predicted Benign", "Predicted Malign"],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=[
                    ["True Benign", "True Malign"],
                    [cm[0][0], cm[1][0]],
                    [cm[0][1], cm[1][1]]
                ]
            )
        ), row=2, col=1
    )

    results.add_trace(
        go.Table(
            header=dict(
                values=[
                    ["Accuracy", accuracy],
                    ["Accuracy Benign", accuracy_benign ],
                    ["Accuracy Malignant", accuracy_malignant]
                ],
                font=dict(size=10),
                align="left"
            )
        ), row=2, col=2
    )

    results.show()