import csv
import os
from shutil import copyfile

trainingFileName = "../res/ISBI2016_ISIC_Part3_Training_GroundTruth.csv"
testingFileName = "../res/ISBI2016_ISIC_Part3_Test_GroundTruth.csv"
trainingBaseFolder = "../res/ISBI2016_ISIC_Part3_Training_Data"
testingBaseFolder = "../res/ISBI2016_ISIC_Part3_Test_Data"


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


if __name__ == "__main__":
    fixFolderStructure(trainingBaseFolder, trainingFileName)
    fixFolderStructure(testingBaseFolder, testingFileName)
