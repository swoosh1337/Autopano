#!/usr/bin/evn python


# Code starts here:

import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
import pickle
import os
import csv
import random
import shutil



def get_patch(img_Original, patchSize, r):
    """
    This function generates a random patch of size patchSize from the image img_Original
    and returns the patch and the 4 corner points of the patch in the Originalinal image
    :param img_Original: Originalinal image
    :param patchSize: Size of the patch
    :param r: Maximum random displacement of the patch
    :return: imgData: Patch of size patchSize x patchSize x 2
    :reurn: H4Pt: homography matrix of size  4 x 2
    """
    
     # Create new folders to save generated data at
    save_Original_data_to = "../Data/Images/Original/"
    if not os.path.exists(save_Original_data_to):
        os.makedirs(save_Original_data_to, exist_ok=True)

    save_warped_data_to = "../Data/Images/Warped/"
    if not os.path.exists(save_warped_data_to):
        os.makedirs(save_warped_data_to, exist_ok=True)

    if not os.path.exists( "../Data/Train/Original/"):
        os.makedirs(  "../Data/Train/Original/",exist_ok=True)

    if not os.path.exists(  "../Data/Train/Warped/",):
        os.makedirs( "../Data/Train/Warped/",exist_ok=True)

    if not os.path.exists( "../Data/Test/Original/"):
        os.makedirs( "../Data/Test/Original/",exist_ok=True)

    if not os.path.exists( "../Data/Test/Warped/"):
        os.makedirs("../Data/Test/Warped/",exist_ok=True)

    if not os.path.exists( "../Data/Val/Original/"):
        os.makedirs("../Data/Val/Original/",exist_ok=True)
        
    if not os.path.exists(  "../Data/Val/Warped/"):
        os.makedirs( "../Data/Val/Warped/",exist_ok=True)
    if len(img_Original.shape) == 3: # if the image is RGB
        img = cv2.cvtColor(img_Original, cv2.COLOR_RGB2GRAY) # convert to grayscale
    else:
        img = img_Original

    #Choose random point as the center of the patch
    x = np.random.randint(patchSize//2+r, img.shape[1]-(patchSize//2+r))  # lower and upper boundaries of the patch
    y = np.random.randint(patchSize//2+r, img.shape[0]-(patchSize//2+r))  # lower and upper boundaries of the patch
    src = []
    #calculate 4 corner points of the patch
    for i in range(4):
        angle = i*np.pi/2 # 0, 90, 180, 270 degrees
        x_offset = int(patchSize//2*np.cos(angle)) # x offset
        y_offset = int(patchSize//2*np.sin(angle)) # y offset
        src.append((x+x_offset, y+y_offset)) # corner points of the patch

 
    #calculate the destination points of the patch
    dst = [(pt[0]+np.random.randint(-r, r), pt[1]+np.random.randint(-r, r)) for pt in src]
   
    H = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst)) # get homography matrix
    H_inv = np.linalg.inv(H) # get inverse homography matrix
    warpImg = cv2.warpPerspective(img, H_inv, (img.shape[1],img.shape[0])) # warp the image using inverse homography matrix
    
    patch1 = img[y-patchSize//2:y+patchSize//2, x-patchSize//2:x+patchSize//2] # get the patch from the Originalinal image
    patch2 = warpImg[y-patchSize//2:y+patchSize//2, x-patchSize//2:x+patchSize//2] # get the patch from the warped image
    imgData = np.dstack((patch1, patch2)) # stack the two patches together# get the homography matrix
    H4Pt = np.subtract(np.array(dst), np.array(src))
    return imgData, H4Pt, patch1, patch2 # return the image and the homography matrix


import os
import random
import shutil

def split_data(path_to_data: str):
    original_data_path = os.path.join(path_to_data, 'Images/Original/')  # path to original images
    warped_data_path = os.path.join(path_to_data, 'Images/Warped/') # path to warped images

    original_file_names = os.listdir(original_data_path) # get all the file names in the original folder
    random.shuffle(original_file_names) # shuffle the file names
    train_size = int(len(original_file_names) * 0.75) # 75% of the data will be used for training
    test_size = int(len(original_file_names) * 0.15) # 15% of the data will be used for testing
    val_size = len(original_file_names) - train_size - test_size # 10% of the data will be used for validation

    train = original_file_names[:train_size] # get the file names for training
    test = original_file_names[train_size:train_size + test_size] # get the file names for testing
    val = original_file_names[train_size + test_size:] # get the file names for validation

    split_data = {
        'Train': train,
        'Test': test,
        'Val': val
    }

    for set_name, file_names in split_data.items():
        original_folder = os.path.join(path_to_data, f'{set_name}/Original/')
        warped_folder = os.path.join(path_to_data, f'{set_name}/Warped/')

        os.makedirs(original_folder, exist_ok=True) # create the folder if it doesn't exist
        os.makedirs(warped_folder, exist_ok=True) # create the folder if it doesn't exist

        for file_name in file_names:
            original_src = os.path.join(original_data_path, file_name) #
            warped_src = os.path.join(warped_data_path, file_name)

            original_dst = os.path.join(original_folder, file_name)
            warped_dst = os.path.join(warped_folder, file_name)

            shutil.move(original_src, original_dst) # move the file from the original folder to the new folder
            shutil.move(warped_src, warped_dst) # move the file from the warped folder to the new folder

    print("Data Organized Successfully!")


trainImg = glob("../Data/Train/*.jpg") # get all the file names in the train folder
valImg = glob("../Data/Val/*.jpg") # get all the file names in the validation folder
Basepath = "../Data/Train/" # path to the train folder
print("Organizing and Splitting Data...")


def data_gen(images, size=(640,480), patch_size=128, r=32):
    """
    Generates the data for training and validation
    :param images: list of images
    :param size: size of the image
    :param patch_size: size of the patch
    :param r: radius of the patch
    :return: X, Y
    """
    X=[]
    Y=[]
    with open("../Data/labels.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        for i in range(len(images)):
            img = plt.imread(images[i])
            img = cv2.resize(img, size)
            img_data, h_data, p1, p2 = get_patch(img, patch_size, r) # get the image and the homography matrix
            cv2.imwrite("../Data/Images/Original/" + str(i + 1) + ".jpg", p1)
            cv2.imwrite("../Data/Images/Warped/" + str(i + 1) + ".jpg", p2)
            X.append(img_data)
            Y.append(h_data)
            img_name = images[i].split("/")[-1]
            data = [img_name, list(np.array(h_data).flatten())] # flatten the homography matrix
            writer.writerow(data) # write the data to the csv file
    return X, Y


X_train=[]
Y_train=[]
X_val = []
Y_val = []

X_train,Y_train = data_gen(trainImg)
# X_val,Y_val = data_gen(valImg)
split_data("../Data")

training = {'features': X_train, 'labels': Y_train}
pickle.dump(training, open("unsupervised.pkl", "wb"))
validation = {'features': X_val, 'labels': Y_val}
pickle.dump(validation, open("validation_less.pkl", "wb"))




def main():
 

    """
    Read a set of images for Panorama stitching
    """

    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

   


if __name__ == "__main__":
    main()

  
