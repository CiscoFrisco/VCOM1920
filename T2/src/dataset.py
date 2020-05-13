import csv
import shutil
import os
import numpy as np
from shutil import copyfile

trainingFileName1 = "../res/Task 1/ISBI2016_ISIC_Part3_Training_GroundTruth.csv"
testingFileName1 = "../res/Task 1/ISBI2016_ISIC_Part3_Test_GroundTruth.csv"
trainingBaseFolder1 = "../res/Task 1/Training"
testingBaseFolder1 = "../res/Tasl 1/Test"

trainingFileName2 = "../res/Task 2/ISIC2018_Task3_Training_GroundTruth.csv"
trainingBaseFolder2 = "../res/Task 2/"


def fixTask2FolderStructure(basefolder, fileName):

    classNames = ["Melanoma",
                  "Melanocytic nevus", "Basal cell carcinoma", "Actinic keratosis", "Benign keratosis",
                  "Dermatofibroma", "Vascular lesion"]

    for className in classNames:
        if not os.path.isdir(basefolder + "/" + className):
            os.mkdir(basefolder + "/" + className)

    with open(fileName, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)

        for row in csvreader:
            imgName = row[0]
            classIndex = row.index("1.0")
            className = classNames[classIndex - 1]

            path = basefolder + "/" + imgName + ".jpg"
            if os.path.exists(path):
                os.rename(path, basefolder + "/" +
                          className + "/" + imgName + ".jpg")


def fixFolderStructure(basefolder, fileName):
    if not os.path.isdir(basefolder + "/benign"):
        os.mkdir(basefolder + "/benign")
        os.mkdir(basefolder + "/malignant")

    with open(fileName, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        for row in csvreader:
            imgName = row[0]
            className = row[1]

            if className == "0.0":
                className = "benign"
            elif className == "1.0":
                className = "malignant"

            path = basefolder + "/" + imgName + ".jpg"

            if os.path.exists(path):
                os.rename(path, basefolder + "/" +
                          className + "/" + imgName + ".jpg")


def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)


def splitTask2Sets(path_to_data, path_to_test_data, train_ratio):
    # get dirs
    _, dirs, _ = next(os.walk(path_to_data))

    # calculates how many train data per class
    data_counter_per_class = np.zeros((len(dirs)))
    for i in range(len(dirs)):
        path = os.path.join(path_to_data, dirs[i])
        files = get_files_from_folder(path)
        data_counter_per_class[i] = len(files)
    test_counter = np.round(data_counter_per_class * (1 - train_ratio))

    # transfers files
    for i in range(len(dirs)):
        path_to_original = os.path.join(path_to_data, dirs[i])
        path_to_save = os.path.join(path_to_test_data, dirs[i])

        # creates dir
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        files = get_files_from_folder(path_to_original)
        # moves data
        for j in range(int(test_counter[i])):
            dst = os.path.join(path_to_save, files[j])
            src = os.path.join(path_to_original, files[j])
            shutil.move(src, dst)


if __name__ == "__main__":
    # fixFolderStructure(trainingBaseFolder1, trainingFileName1)
    # fixFolderStructure(testingBaseFolder1, testingFileName1)
    fixTask2FolderStructure(trainingBaseFolder2, trainingFileName2)
    splitTask2Sets("../res/Task 2/Training", "../res/Task 2/Test", 0.7)
