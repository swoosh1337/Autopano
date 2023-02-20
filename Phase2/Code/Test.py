


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
from Network.Network_supervised import HomographyModel
from Network.Network_supervised import Net
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch

# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize


def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    return Img


def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    I1 = Img

    if I1 is None:
        # OpenCV returns empty list if image is not read!
        print("ERROR: Image I1 cannot be read")
        sys.exit()

    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1


def TestOperation(TestSet, ImageSize, ModelPath, LabelsPathPred):



    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    model = HomographyModel()

    print("----------------------------------------")

    CheckPoint = torch.load(ModelPath, map_location=torch.device('cpu'))

    print(
        "Number of parameters in this model are %d " % len(model.state_dict().items())
    )

    OutSaveT = open(LabelsPathPred, "w")

    for count in tqdm(range(len(TestSet))):

        Img, Label = TestSet[count]
        Img, ImgOrg = ReadImages(Img)
        dim0 = ImgOrg.shape[0]
        dim1 = ImgOrg.shape[1]
        dim2 = ImgOrg.shape[2]
        ImgOrg = ImgOrg.view(1, dim0, dim1, dim2)
        stacked_images = np.float32(np.concatenate([Img, ImgOrg], axis=3))

        # Get label
        # Append All Images and Mask

        PredT = torch.argmax(model(stacked_images)).item()
        OutSaveT.write(str(PredT) + "\n")
    OutSaveT.close()


def Accuracy(Pred, GT):
    """
    Inputs:
    Pred are the predicted labels
    GT are the ground truth labels
    Output:
    Accuracy in percentage
    """
    return np.sum(np.array(Pred) == np.array(GT)) * 100.0 / len(Pred)


def ReadLabels(LabelsPathTest, LabelsPathPred):
    if not (os.path.isfile(LabelsPathTest)):
        print("ERROR: Test Labels do not exist in " + LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, "r")
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if not (os.path.isfile(LabelsPathPred)):
        print("ERROR: Pred Labels do not exist in " + LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, "r")
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())

    return LabelTest, LabelPred


def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """
    print(LabelsTrue, LabelsPred)
    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    LabelsTrue = [float(x) for x in LabelsTrue]
    LabelsPred = [float(x) for x in LabelsPred]
    cm = confusion_matrix(
        y_true=LabelsTrue, y_pred=LabelsPred  # True class for test-set.
    )  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + " ({0})".format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print("Accuracy: " + str(Accuracy(LabelsPred, LabelsTrue)), "%")



def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="../Checkpoints/99model.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default="../Data/Test/",
        help="Path to load images from, Default:BasePath",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default="../Code/TxtFiles/LabelsTest1.txt",
        help="Path of labels file, Default:..Code/TxtFiles/LabelsTest.txt"
    )
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath

    # Setup all needed parameters including file reading
    ImageSize = SetupAll()

    # Define PlaceHolder variables for Input and Predicted output
    # ImgPH = tf.placeholder("float", shape=(1, ImageSize[0], ImageSize[1], 3))
    ImgPH = "../Data/Test/Orig/*.jpg"
    ImgPHW = "../Data/Test/Warped/*.jpg"
    LabelsPathPred = "./TxtFiles/LabelsPred.txt"  # Path to save predicted labels

    import glob
    TestSet = []
    images_orig = glob.glob(ImgPH)
    images_warp = glob.glob(ImgPHW)

    for img in images_orig:
        orig_img = cv2.imread(img)

        warp_img = cv2.imread(img.replace("Orig", "Warped"))


        orig_img = orig_img.astype(np.float32)
        warp_img = warp_img.astype(np.float32)

        orig_img = torch.tensor(orig_img)
        warp_img = torch.tensor(warp_img)

        tup = (orig_img, warp_img)
        TestSet.append(tup)




    TestOperation(TestSet, ImageSize, ModelPath, LabelsPathPred)



    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    ConfusionMatrix(LabelsTrue, LabelsPred)


if __name__ == "__main__":
    main()