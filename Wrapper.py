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

import numpy as np
import cv2
import copy
import argparse
import glob
import matplotlib.pyplot as plt
import os
# Add any python libraries here


def anms(cornerMap, Nbest):
    """
    Function implements Adaptive Non-Maximal Suppression (ANMS) algorithm
    :param cornerMap: Corner response map
    :param Nbest: Number of best corners to be returned
    :return: List of Nbest corners
    """
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
    print()
    
    return z[:Nbest]

def featureDescriptor(image, map):
    """
    Function implements feature descriptor
    :param image: Input image
    :param map: List of corners
    :return: List of feature vectors"""
    mean_img = np.mean(image)
    std_img = np.std(image)
    ymax, xmax = image.shape
    listofVectors = []
    listxy = []
    for xi, yi in map:
        xlower = max(0, xi - 20)
        xupper = min(xmax, xi + 20)
        ylower = max(0, yi - 20)
        yupper = min(ymax, yi + 20)
        imagepatch = image[ylower:yupper + 1, xlower:xupper + 1]
        imagepatch = (imagepatch - mean_img) / std_img
        blurredOutput = cv2.GaussianBlur(imagepatch, (5, 5), 1)
        subsampledOutput = cv2.resize(blurredOutput, (8, 8))
        vector = subsampledOutput.ravel()
        mean_vec = np.mean(vector)
        std_vec = np.std(vector)
        vector = (vector - mean_vec) / std_vec
        listxy.append((xi, yi))
        listofVectors.append(vector)
    return listofVectors, listxy


def featureMatching(vectorlist1, listxy1, vectorlist2, listxy2, threshold):
    """
    Function implements feature matching
    :param vectorlist1: List of feature vectors for image 1
    :param listxy1: List of corners for image 1
    :param vectorlist2: List of feature vectors for image 2
    :param listxy2: List of corners for image 2
    :param threshold: Threshold for feature matching
    :return: List of matched corners
    """
    pair1 = []
    pair2 = []
    distances = []
    for vi, (xi1, yi1) in zip(vectorlist1, listxy1):
        diff = vectorlist2 - vi
        sq_diff = diff**2
        sum_sq_diff = np.sum(sq_diff, axis=1)
        min1 = np.min(sum_sq_diff)
        pos1 = np.argmin(sum_sq_diff)
        sum_sq_diff[pos1] = np.inf
        min2 = np.min(sum_sq_diff)
        if min1 / min2 > threshold:
            continue
        pair1.append((xi1, yi1))
        pair2.append(listxy2[pos1])
        distances.append(min1)
    return np.array(pair1, dtype=np.float32), np.array(pair2, dtype=np.float32), np.array(distances, dtype=np.float32)


def draw_matches(keypoints1,keypoints2):
    """"
    Function to draw matches between two images
    :param keypoints1: List of keypoints for image 
    :param keypoints2: List of keypoints for image
    :return: List of matches
    """
    matchlist = []
    for i in range(len(keypoints1)):
        p1 = np.array(keypoints1[i])
        p2 = np.array(keypoints2[i])
        matchlist.append(cv2.DMatch(i, i, float(np.linalg.norm(p1 - p2))))
    return matchlist

def keypts(kp):
    """
    Function to convert keypoints to list of tuples
    :param kp: List of keypoints
    :return: List of tuples
    """
    return [cv2.KeyPoint(i[0], i[1], 3) for i in kp]

def ransac(pointlist1, pointlist2, threshhold, Nmax, percentage):
    """"
    Function to implement RANSAC algorithm
    :param pointlist1: List of points for image 1
    :param pointlist2: List of points for image 2
    :param threshhold: Threshold for RANSAC
    :param Nmax: Maximum number of iterations
    :param percentage: Percentage of inliners
    :return: Homography matrix
    """
    idx = []
    totalinliners = 0
    updatedH = np.zeros((3,3))
    for _ in range(Nmax):
        fourfeatures1 = []
        fourfeatures2 = []
        for i in range(4):
            idx = np.random.randint(0, len(pointlist1))
            fourfeatures1.append(tuple(pointlist1[idx]))
            fourfeatures2.append(tuple(pointlist2[idx]))
        H = cv2.getPerspectiveTransform(np.float32(fourfeatures1), np.float32(fourfeatures2))
        inlinersindexlist = []
        for i in range(len(pointlist1)):
            p_dash = np.matmul(H, [pointlist1[i][0], pointlist1[i][1], 1])
            if p_dash[2] == 0:
                p_dash[2] = 0.000001
            px = np.float32(p_dash[0]/p_dash[2])
            py = np.float32(p_dash[1]/p_dash[2])
            point = np.array((pointlist2[i][0], pointlist2[i][1]))
            p_dash = np.array((px, py))
            ssd = np.sum((point-p_dash)**2)
            if ssd < threshhold:
                inlinersindexlist.append(i)
        if totalinliners < len(inlinersindexlist):
            totalinliners = len(inlinersindexlist)
            z = copy.deepcopy(inlinersindexlist)
            updatedH = H
            if len(inlinersindexlist) > percentage*len(pointlist1):
                break
    return z,updatedH

def ransac_executor(image1pts, image2pts, distancelist, threshhold, Nmax, percentage):
    """
    Function to execute RANSAC algorithm
    :param image1pts: List of points for image 1
    :param image2pts: List of points for image 2
    :param distancelist: List of distances between points
    :param threshhold: Threshold for RANSAC
    :param Nmax: Maximum number of iterations
    :param percentage: Percentage of inliners
    :return: List of keypoints, list of matches, homography matrix, flag
    """

    idx, h = ransac(image1pts, image2pts, threshhold, Nmax, percentage)
    keypoints1 = []
    keypoints2 = []
    matchlist = []
    for i in idx:
        keypoints1.append(tuple(image1pts[i]))
        keypoints2.append(tuple(image2pts[i]))
        matchlist.append(distancelist[i])
    kptconverted1 = keypts(keypoints1)
    kptconverted2 = keypts(keypoints2)
    distancesconverted = draw_matches(keypoints1,keypoints2)
    print(len(keypoints1),len(keypoints2))
    if len(keypoints1) < 10 :
        flag = False
    else:
        flag = True

    return kptconverted1, kptconverted2, distancesconverted, h, flag

def stitch_img(img1, H, shape):
    """
    Function to stitch two images
    :param img1: Image 1
    :param H: Homography matrix
    :param shape: Shape of image 2
    :return: Stitched image
    """
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = shape[:2]
    list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])
    output_img = cv2.warpPerspective(img1, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    origin_offset_x = x_min
    origin_offset_y = y_min
    return output_img, origin_offset_x, origin_offset_y

def getH(img1_loc, img2_loc):
    """
    Function to get homography matrix
    :param img1_loc: Image 1 location
    :param img2_loc: Image 2 location
    :return: Homography matrix
    """
    imgset = []
    for img in [img1_loc, img2_loc]:
        
        imgset.append(img)
    
    vectorbank = []
    positionbank = []
    for img in imgset[:2]:
        img1 = copy.deepcopy(img)
        gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        cornerMap = cv2.goodFeaturesToTrack(gray_img, 10000, 0.001, 15)
        cornerMap = np.int32(cornerMap)

        anmsMap = anms(cornerMap, 1000)
        img2 = copy.deepcopy(img)
        for point in anmsMap:
            x, y = point
            cv2.circle(img2, (x, y), 2, 255, -1)

        img3 = copy.deepcopy(gray_img)
        vl, pl = featureDescriptor(img3, anmsMap)
        vectorbank.append(np.array(vl, dtype=np.float32))
        positionbank.append(np.array(pl, dtype=np.float32))

    image1pointlist, image2pointlist, distancelst = featureMatching(vectorbank[0], positionbank[0], vectorbank[1], positionbank[1], 0.9)
    pointmatches = draw_matches(image1pointlist, image2pointlist)
    kpt1 = keypts(image1pointlist)
    kpt2 = keypts(image2pointlist)
    outputimg1 = np.array([])
    matchedImage = cv2.drawMatches(imgset[0], kpt1, imgset[1], kpt2, pointmatches, outputimg1, 1)
    cv2.imshow('matching', matchedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    outputimg2 = np.array([])
    k1, k2, mimg, h ,flag = ransac_executor(image1pointlist, image2pointlist, distancelst, threshhold=150, Nmax=5000, percentage= 0.9)
    print(k1, k2, mimg, h, flag, "<--ransac")
    ransacedImage = cv2.drawMatches(imgset[0], k1, imgset[1], k2, mimg, outputimg2, 0)
    cv2.imshow('ransac', ransacedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return h, flag



# Stitch and Blend the Panorama
def Blend(images):
    """
    Function to blend the images
    :param images: List of images
    :return: Blended image
    """
    n = len(images)
     # Set the desired shape here
    desired_shape = (500, 500)
    img1 = images[0]

    
    for im in images[1:]:
        # desired_shape = img1.shape 
        # img1 = cv2.resize(img1, (desired_shape[1], desired_shape[0]))
        # if im.shape != desired_shape:
        #     im = cv2.resize(im, (desired_shape[1], desired_shape[0]))
       
        
        # H = homography(img1, im, bff_match=False)
        H,flag = getH(img1,im)
        if flag == False:
            print('Number of matches is less than required')
            break
        # Hinv = np.linalg.inv(H)
        try:
            imgholder, origin_offset_x,origin_offset_y = stitch_img(img1,H,im.shape)
        except cv2.error as e:
            print("OpenCV Error:", e)
            break
        oX = abs(origin_offset_x)
        oY = abs(origin_offset_y)
        for y in range(oY,im.shape[0]+oY):
            for x in range(oX,im.shape[1]+oX):
                img2_y = y - oY
                img2_x = x - oX
                imgholder[y,x,:] = im[img2_y,img2_x,:]
        img1 = imgholder
    # resize_pano = cv2.resize(img1,[1280,1024])
    cv2.imshow('Pano', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('pano.jpeg',img1)
    return img1



def main():
    
    # # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath',default='../Data/Train/',help='Base path to get images from,Default:../Data/Train/')
    Parser.add_argument('--TestSet',default='../Data/Test/TestSet1/',help='Test set to run algorithm on,Default:Set1')
    """
    Read a set of images for Panorama stitching
    """
    Args = Parser.parse_args()
    img_set = Args.TestSet
    images = [cv2.imread(file) for file in sorted(glob.glob(img_set+'/*.jpg'))]
   
    Blend(images)
   


if __name__ == "__main__":
    main()