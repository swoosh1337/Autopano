# #!/usr/bin/evn python

# """
# RBE/CS Fall 2022: Classical and Deep Learning Approaches for
# Geometric Computer Vision
# Project 1: MyAutoPano: Phase 2 Starter Code


# Author(s):
# Lening Li (lli4@wpi.edu)
# Teaching Assistant in Robotics Engineering,
# Worcester Polytechnic Institute
# """


# # Code starts here:

# import numpy as np
# import cv2
# from glob import glob
# import matplotlib.pyplot as plt
# import pickle




# def getPatch(img_orig, patchSize, r):
#     """
#     This function generates a random patch of size patchSize from the image img_orig
#     and returns the patch and the 4 corner points of the patch in the original image
#     :param img_orig: Original image
#     :param patchSize: Size of the patch
#     :param r: Maximum random displacement of the patch
#     :return: imgData: Patch of size patchSize x patchSize x 2
#     :reurn: H4Pt: homography matrix of size  4 x 2
#     """
#     if len(img_orig.shape) == 3: # if the image is RGB
#         img = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY) # convert to grayscale
#     else:
#         img = img_orig

#     #Choose random point as the center of the patch
#     x = np.random.randint(patchSize//2+r, img.shape[1]-(patchSize//2+r))  # lower and upper boundaries of the patch
#     y = np.random.randint(patchSize//2+r, img.shape[0]-(patchSize//2+r))  # lower and upper boundaries of the patch
#     src = []
#     #calculate 4 corner points of the patch
#     for i in range(4):
#         angle = i*np.pi/2 # 0, 90, 180, 270 degrees
#         x_offset = int(patchSize//2*np.cos(angle)) # x offset
#         y_offset = int(patchSize//2*np.sin(angle)) # y offset
#         src.append((x+x_offset, y+y_offset)) # corner points of the patch

 
#     #calculate the destination points of the patch
#     dst = [(pt[0]+np.random.randint(-r, r), pt[1]+np.random.randint(-r, r)) for pt in src]
   
#     H = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst)) # get homography matrix
#     H_inv = np.linalg.inv(H) # get inverse homography matrix
#     warpImg = cv2.warpPerspective(img, H_inv, (img.shape[1],img.shape[0])) # warp the image using inverse homography matrix
    
#     patch1 = img[y-patchSize//2:y+patchSize//2, x-patchSize//2:x+patchSize//2] # get the patch from the original image
#     patch2 = warpImg[y-patchSize//2:y+patchSize//2, x-patchSize//2:x+patchSize//2] # get the patch from the warped image
#     imgData = np.dstack((patch1, patch2)) # stack the two patches together# get the homography matrix
#     H4Pt = np.subtract(np.array(dst), np.array(src))
#     return imgData, H4Pt # return the image and the homography matrix



# trainImg = glob("../Data/Train/*.jpg")
# valImg = glob("../Data/Val/*.jpg")
# Basepath = "../Data/Train/"
# print("No. of Training Images = " + str(len(trainImg)))
# print("No. of Validation Images = " + str(len(valImg)))



# def dataGen(Image,size=(640,480),patchSize=128,r=32):
#     X=[]
#     Y=[]
#     for j in range(1):
#         print("No. of samples collected = "+str(len(Image)*j))
#         for i in range(len(Image)):  
#             img = plt.imread(Image[i])
#             img = cv2.resize(img,size)
#             imgData,hData = getPatch(img,patchSize,r)
#             X.append(imgData)
#             Y.append(hData)
#     return X,Y


# X_train=[]
# Y_train=[]
# X_val = []
# Y_val = []

# X_train,Y_train = dataGen(trainImg)
# X_val,Y_val = dataGen(valImg)

# training = {'features': X_train, 'labels': Y_train}
# pickle.dump(training, open("unsupervised.pkl", "wb"))
# validation = {'features': X_val, 'labels': Y_val}
# pickle.dump(validation, open("validation_less.pkl", "wb"))

# # Add any python libraries here


# def main():
#     # Add any Command Line arguments here
#     # Parser = argparse.ArgumentParser()
#     # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

#     # Args = Parser.parse_args()
#     # NumFeatures = Args.NumFeatures

#     """
#     Read a set of images for Panorama stitching
#     """

#     """
# 	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
# 	"""

#     """
# 	Image Warping + Blending
# 	Save Panorama output as mypano.png
# 	"""

   


# if __name__ == "__main__":
#     main()




import argparse
import csv
import os
from random import randint

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--mscoco_data_path", type=str, help="Abs. path to images for synthetic data generation")
args = parser.parse_args()

MSCOCO_DATA_PATH = args.mscoco_data_path
if not MSCOCO_DATA_PATH:
    MSCOCO_DATA_PATH = '../Data/Train/'


def generate_data(patch_size: int = 128, perturb_max: int = 32, pixel_buffer_len: int = 150, color: bool = True):
  
    # Create new folders to save generated data at
    save_orig_data_to = "../Data/Raw/Orig/"
    if not os.path.exists(save_orig_data_to):
        os.makedirs(save_orig_data_to, exist_ok=True)

    save_warped_data_to = "../Data/Raw/Warped/"
    if not os.path.exists(save_warped_data_to):
        os.makedirs(save_warped_data_to, exist_ok=True)

    stride = int(0.25 * patch_size)
    rho_range = [-perturb_max, perturb_max]

    images = os.listdir(MSCOCO_DATA_PATH)
    img_paths = [MSCOCO_DATA_PATH + image for image in images]

    with open(os.path.join('../Data', 'labels.csv'), 'w') as labels_file:
        writer = csv.writer(labels_file)

        for img_path in img_paths:
            img_name = img_path.split('/')[-1].split('.')[0]  # The index of image in MSCOCO dataset given

            img = cv2.imread(img_path)
            if not color:
                # Convert to grayscale
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Create bounds of active region
            h, w = img.shape[0], img.shape[1]
            active_h = [pixel_buffer_len, h - pixel_buffer_len]
            active_w = [pixel_buffer_len, w - pixel_buffer_len]

            image_counter = 0
            for i in range(active_h[0], active_h[1], stride):
                if active_h[1] - i < patch_size:  # Ignore dimensions that give poor data
                    continue

                for j in range(active_w[0], active_w[1], stride):
                    if active_w[1] - j < patch_size:  # Ignore dimensions that give poor data
                        continue

                    # Get original and perturbed corners
                    patch_corners = np.array(
                        [(j, i), (j + patch_size, i), (j, i + patch_size), (j + patch_size, i + patch_size)]
                    )
                    perturbations = [(randint(*rho_range), randint(*rho_range)) for _ in range(4)]  # H4pt
                    perturbed_corners = patch_corners + perturbations

                    # Estimate homography between the two sets of corners
                    homograhy, _ = cv2.findHomography(patch_corners, perturbed_corners)
                    transformed_img = cv2.warpPerspective(img, np.linalg.inv(homograhy), (h, w), flags=cv2.INTER_LINEAR)

                    # Get the corresponding original & perturbed patches
                    img_crop = img[i: i + patch_size, j: j + patch_size]
                    transformed_img_crop = transformed_img[i: i + patch_size, j: j + patch_size]
                    if transformed_img_crop.shape != img_crop.shape:
                        continue

                    image_counter += 1
                    patch_name = f'{img_name}_{image_counter}.jpg'
                    cv2.imwrite(os.path.join(save_orig_data_to, "orig_" + patch_name), img_crop)
                    cv2.imwrite(os.path.join(save_warped_data_to + "warped_" + patch_name), transformed_img_crop)

                    writer.writerow([patch_name, list(np.array(perturbations).flatten())])


def organize_shuffle_split_data(path_to_data: str):
    """
    Note: Assumes 'path_to_data' contains 'Raw' image folder.
    Takes the generated data, shuffles it, splits and organises into Train(75%), Test(15%) &
    Validation(10%) sets
    """
    from shutil import move
    from sklearn.model_selection import train_test_split

  

    orig_data_path = os.path.join(path_to_data, 'Raw/Orig/')
    warped_data_path = os.path.join(path_to_data, 'Raw/Warped/')

    os.makedirs(path_to_data + "/Train/Orig/")
    os.makedirs(path_to_data + "/Train/Warped/")
    os.makedirs(path_to_data + "/Test/Orig/")
    os.makedirs(path_to_data + "/Test/Warped/")
    os.makedirs(path_to_data + "/Val/Orig/")
    os.makedirs(path_to_data + "/Val/Warped/")

    orig_file_names = os.listdir(orig_data_path)
    train, test = train_test_split(orig_file_names, test_size=0.25, random_state=45)
    test, val = train_test_split(test, test_size=0.4, random_state=45)

    for orig_name in train:
        raw_name = orig_name[5:]  # raw_name for 'orig_2600_13.jpg' is '2600_13.jpg'
        warped_name = "warped_" + raw_name

        # Move train images into respective folders
        move(
            os.path.join(orig_data_path, orig_name),
            os.path.join(path_to_data + f"/Train/Orig/{raw_name}")
        )
        move(
            os.path.join(warped_data_path, warped_name),
            os.path.join(path_to_data + f"/Train/Warped/{raw_name}")
        )

    for orig_name in test:
        raw_name = orig_name[5:]  # raw_name for 'orig_2600_13.jpg' is '2600_13.jpg'
        warped_name = "warped_" + raw_name

        # Move train images into respective folders
        move(
            os.path.join(orig_data_path, orig_name),
            os.path.join(path_to_data + f"/Test/Orig/{raw_name}")
        )
        move(
            os.path.join(warped_data_path, warped_name),
            os.path.join(path_to_data + f"/Test/Warped/{raw_name}")
        )

    for orig_name in val:
        raw_name = orig_name[5:]  # raw_name for 'orig_2600_13.jpg' is '2600_13.jpg'
        warped_name = "warped_" + raw_name

        # Move train images into respective folders
        move(
            os.path.join(orig_data_path, orig_name),
            os.path.join(path_to_data + f"/Val/Orig/{raw_name}")
        )
        move(
            os.path.join(warped_data_path, warped_name),
            os.path.join(path_to_data + f"/Val/Warped/{raw_name}")
        )


if __name__ == '__main__':
    generate_data()
    organize_shuffle_split_data("../Data")

