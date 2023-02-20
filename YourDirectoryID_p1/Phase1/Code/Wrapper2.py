#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:
import os
import numpy as np
import cv2
import copy
import argparse
import glob
import matplotlib.pyplot as plt
from os import path
# Add any python libraries here

def anms(cornerMap, Nbest):
    cornerMap = np.squeeze(cornerMap)
    lc = {}
    for i in range(1, len(cornerMap)):
        dist = cornerMap[:i] - cornerMap[i]
        dist = dist * dist
        dist = np.sum(dist, axis=1)
        mindist = np.min(list(dist))
        idx, = np.where(dist==mindist)
        location = cornerMap[idx]
        x, y = location[0]
        location = (x, y)
        if location in lc:
            if mindist > lc[location]:
                continue
        lc.update({location : mindist})
    sorted_lc = dict(sorted(lc.items(), key=lambda a:a[1]))
    z = list(sorted_lc.keys())
    z.reverse()
    return z[:Nbest]


def featureDescriptor(image, map):
    ymax, xmax = image.shape
   
    listofVectors = []
    for i in range(len(map)):
        xi, yi = map[i]
        print(xi, yi)
        xlower = xi - 20
        xupper = xi + 20
        ylower = yi - 20
        yupper = yi + 20
        if xi < 20:
            print("yay1")
            xlower = 0
            xupper = xi + 20
        if xi >= (xmax - 20):
            print("yay2")
            xlower = xi - 20
            xupper = xmax
        if yi < 20:
            print("yay3")
            ylower = 0
            yupper = yi + 20
        if xi >= (xmax - 20):
            print("yay4")
            ylower = yi - 20
            yupper = xmax
        print(xlower, xupper, ylower, yupper)
        imagepatch = image[ylower:yupper + 1, xlower:xupper + 1]
        blurredOutput = cv2.GaussianBlur(imagepatch, (5, 5), 1)
        subsampledOutput = cv2.resize(blurredOutput, (8, 8))
        vector = subsampledOutput.ravel()
        vector = (vector - np.mean(vector))/np.std(vector)
        listofVectors.append(vector,(xi, yi))
    return listofVectors


def featureMatching(vectorlist1, vectorlist2):
     for i in vectorlist1:
         distance = vectorlist2 - i
         sq_distance = distance * distance
         sum = np.sum(sq_distance, axis=1)
         max = np.max(sum)
         pos1 = sum.index(max)
         sum.remove(max)
         max2 = np.max(sum)



def main():
    # Add any Command Line arguments here
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--Path', default=str(parent_dir)+"/Data/Train/Set1", help = 'Input Image Location')
    Args = Parser.parse_args()
    NumFeatures = Args.NumFeatures
    Path = Args.Path
    """
    Read a set of images for Panorama stitching
    """
    img_location = Path + "/*.jpg"
    imgset = []
    for filename in glob.glob(img_location):
        z = cv2.imread(filename)
        imgset.append(z)

    


    """
	Corner Detection
	Save Corner detection output as corners.png
	"""

    # for img in imgset:
    #     img1 = copy.deepcopy(img)
    #     gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #     cornerMap = cv2.goodFeaturesToTrack(gray_img, 1000, 0.01, 5)
    #     cornerMap = np.int32(cornerMap)
    #     for point in cornerMap:
    #         x, y = point.reshape(-1)
    #         cv2.circle(img1, (x, y), 3, 255, -1)
    #     plt.imshow(img1)
    #     plt.show()

    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""


    # anmsMap = anms(cornerMap, 200)
    # print(anmsMap)
    # # print(len(anmsMap))
    # for point in anmsMap:
    #     x, y = point
    #     cv2.circle(img, (x, y), 3, 255, -1)
    # plt.imshow(img)
    # plt.show()


    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""

   
    image1 = imgset[0]
    image2 = imgset[1]
    gray_img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    cornerMap1 = cv2.goodFeaturesToTrack(gray_img1, 1000, 0.01, 5)
    cornerMap1 = np.int32(cornerMap1)
    cornerMap2 = cv2.goodFeaturesToTrack(gray_img2, 1000, 0.01, 5)
    cornerMap2 = np.int32(cornerMap2)

    best1 = anms(cornerMap1, 200)
    best2 = anms(cornerMap2, 200)
   

    vector1 = featureDescriptor(gray_img1, best1)
    vector2 = featureDescriptor(gray_img2, best2)






# Initialize empty list to store matched keypoints
    matched_keypoints = []

    for i in range(len(vector1)):
        # Initialize variables to store best and second best match
        best_match = None
        second_best_match = None
        best_distance = float('inf')
        second_best_distance = float('inf')

    #     # Iterate through keypoints in image 2
        for j in range(len(vector1)):
            # Compute sum of square differences between feature vectors
            distance = np.sum((vector1[i] - vector2[j])**2)
            # Update best and second best match if necessary
            if distance < best_distance:
                second_best_distance = best_distance
                second_best_match = best_match
                best_distance = distance
                best_match = j
            elif distance < second_best_distance:
                second_best_distance = distance
                second_best_match = j

        # Compute ratio of best match to second best match
        ratio = best_distance / second_best_distance

        # If ratio is below some threshold, store matched keypoints
        if ratio < 0.9:
            matched_keypoints.append((i, best_match))

    # Compute homography using matched keypoints


        print(matched_keypoints)
#         kp1, des1 = matched_keypoints[:,0:2], matched_keypoints[:,2:]
#         kp2, des2 = matched_keypoints[:,0:2], matched_keypoints[:,2:]   

#         bf = cv2.BFMatcher()

# # Match the descriptors
#         matches = bf.match(des1, des2)

# # Draw the matches
#         img3 = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=2)

#         cv2.imshow("Image", img3)
#         cv2.waitKey(0)
        # homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # # Draw matches
        # img_matches = cv2.drawMatches(image1, matched_keypoints[i], image2, matched_keypoints[j], matched_keypoints, None)
        # cv2.imshow('Matched Features', img_matches)
        # cv2.waitKey(0)


    """
	Refine: RANSAC, Estimate Homography
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
