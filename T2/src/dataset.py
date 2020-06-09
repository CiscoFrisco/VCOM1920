import csv
import shutil
import os
import sys
import numpy as np
from shutil import copyfile

trainingFileName1 = "../res/Task 1/ISBI2016_ISIC_Part3_Training_GroundTruth.csv"
testingFileName1 = "../res/Task 1/ISBI2016_ISIC_Part3_Test_GroundTruth.csv"
trainingBaseFolder1 = "../res/Task 1/Training"
testingBaseFolder1 = "../res/Task 1/Test"

trainingFileName2 = "../res/Task 2/ISIC2018_Task3_Training_GroundTruth.csv"
trainingBaseFolder2 = "../res/Task 2/Training"
testingBaseFolder2 = "../res/Task 2/Test"

# Creates a folder for the bening and malignant classes, and moves the images to their appropriate place
def fixTask1FolderStructure(basefolder, fileName):
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

# Creates a folder for each of the 7 classes, and moves the images to their appropriate place
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
   


def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)

# Splits a set in two, while keeping the proportions of each class
# Useful two split the dataset for the second exercise in training and testing
def splitSets(path_to_data, path_to_test_data, train_ratio):
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


def main():
    if len(sys.argv) != 2 or (sys.argv[1] != '1' and sys.argv[1] != '2'):
        print("Usage: python " + sys.argv[0] + " <TASK>")
        print("Where TASK is one of 1 or 2.")
        return

    task = sys.argv[1]

    if task == '1':
        fixTask1FolderStructure(trainingBaseFolder1, trainingFileName1)
        fixTask1FolderStructure(testingBaseFolder1, testingFileName1)
    else:
        fixTask2FolderStructure(trainingBaseFolder2, trainingFileName2)
        splitSets(trainingBaseFolder2, testingBaseFolder2, 0.7)

if __name__ == "__main__":
    main()
