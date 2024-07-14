import cv2
import numpy as np

# Real-world coordinates (for img1)
world_points = np.array([
    [11, 10],
    [-7, 7],
    [-3,-14],
    [5, -2]
], dtype=np.float32)

# Image coordinates (for img1)
image_points = np.array([
    [2280, 1998],
    [929, 2171],
    [1195, 3820],
    [1828, 2873]
], dtype=np.float32)


# Compute the homography matrix
H, _ = cv2.findHomography(world_points, image_points)

print("Homography matrix:")
print(H)

# Compute the inverse homography matrix
H_inv = np.linalg.inv(H)

print("Inverse Homography matrix:")
print(H_inv)


# ========== Real-world coordinates to test ==========
# 1
new_world_point = np.array([[-2, 10]], dtype=np.float32)
# 2
# new_world_point = np.array([[-10, -17]], dtype=np.float32)

new_world_point_homogeneous = np.hstack([new_world_point, np.ones((new_world_point.shape[0], 1))])
print('new_world_point: ', new_world_point_homogeneous)

# Apply the homography matrix using matrix multiplication
new_image_point_homogeneous = new_world_point_homogeneous @ H.T

# Convert back to Cartesian coordinates
new_image_point = new_image_point_homogeneous[:, :2] / new_image_point_homogeneous[:, 2, np.newaxis]

print("New image point (pixels): ", new_image_point)


# ========== Pixel coordinates to test ==========

# New pixel point to test
new_image_pixel = np.array([[1331, 1955]], dtype=np.float32)
new_image_pixel_homogeneous = np.hstack([new_image_pixel, np.ones((new_image_pixel.shape[0], 1))])
print('new_image_pixel: ', new_image_pixel_homogeneous)

# Apply the inverse homography matrix
new_world_point_homogeneous_inv = new_image_pixel_homogeneous @ H_inv.T

# Convert back to Cartesian coordinates
new_world_point_inv = new_world_point_homogeneous_inv[:, :2] / new_world_point_homogeneous_inv[:, 2, np.newaxis]

print("Real-world coordinates from pixel point:")
print(new_world_point_inv)