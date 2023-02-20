import matplotlib.pyplot as plt
import numpy as np
import cv2

def ransac_homography(points1, points2, Nmax=1000, threshold=1):
    """
    Estimates a Homography matrix using the RANSAC algorithm.
    Parameters:
        - points1: List of N feature points in image 1 (Nx2 array)
        - points2: List of N feature points in image 2 (Nx2 array)
        - Nmax: Maximum number of iterations
        - threshold: Maximum distance between a point and its corresponding epipolar line
    Returns:
        - H: Estimated Homography matrix (3x3 array)
        - inliers: Indices of inliers in the input points (Nx1 array)
    """
    # Number of points
    N = points1.shape[0]
    # Best Homography matrix and inliers
    H_best = None
    inliers_best = None
    # Iterations
    for i in range(Nmax):
        # Randomly select 4 feature points
        sample = np.random.choice(N, 4, replace=False)
        # Compute Homography matrix
        H = cv2.getPerspectiveTransform(points1[sample], points2[sample])
        # Compute inliers
        inliers = np.where(np.sum((points2 - np.matmul(H, np.hstack((points1, np.ones((N, 1)))).T).T) ** 2, axis=1) < threshold ** 2)[0]
        # Update best Homography matrix and inliers
        if inliers.shape[0] > inliers_best.shape[0]:
            H_best = H
            inliers_best = inliers
    # Re-compute least-squares Homography matrix estimate on all of the inliers
    H, _ = cv2.findHomography(points1[inliers_best], points2[inliers_best])
    return H, inliers_best

# Generate some sample data
points1 = np.random.rand(10, 2)
points2 = np.random.rand(10, 2)
# Add some noise to the points
points1 += np.random.randn(10, 2) * 0.1
points2 += np.random.randn(10, 2) * 0.1
# Apply an arbitrary Homography transformation to the points
H_true = np.array([[1, 0, 0.5], [0, 1, 0.1], [0, 0, 1]])
points2 = np.matmul(H_true, np.hstack((points1, np.ones((10, 1)))).T).T[:, :2]
# Add some outliers to the points
points2[:2] += np.random.randn(2, 2) * 0.5

# Estimate Homography matrix using RANSAC
H, inliers = ransac_homography(points1, points2, Nmax=1000, threshold=0.1)

# Plot the matched keypoints
plt.scatter(points1[:, 0], points1[:, 1], color='b')
plt.scatter(points2[:, 0], points2[:, 1], color='r')
for i in range(points1.shape[0]):
    plt.plot([points1[i, 0], points2[i, 0]], [points1[i, 1], points2[i, 1]], 'g-')
plt.show()

# Plot the inliers
plt.scatter(points1[inliers, 0], points1[inliers, 1], color='b')
plt.scatter(points2[inliers, 0], points2[inliers, 1], color='r')
for i in range(points1[inliers].shape[0]):
    plt.plot([points1[inliers][i, 0], points2[inliers][i, 0]], [points1[inliers][i, 1], points2[inliers][i, 1]], 'g-')
plt.show()
