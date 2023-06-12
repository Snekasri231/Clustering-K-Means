import cv2
import numpy as np

# Load the two images
img1 = cv2.imread('Pillar1.png')
img2 = cv2.imread('Pillar2.png')

# Convert the images to the L*a*b color space
lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

# Flatten the images to a one-dimensional array
flat1 = lab1.reshape((-1, 3))
flat2 = lab2.reshape((-1, 3))

# Apply K-Means clustering to the flattened images
k = 2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
compactness, labels, centers = cv2.kmeans(np.float32(flat1), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
compactness, labels, centers = cv2.kmeans(np.float32(flat2), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Create a mask for the pixels that belong to the first cluster in both images
mask1 = labels.ravel() == 0
mask2 = labels.ravel() == 0

# Calculate the absolute difference between the two images
diff = np.abs(flat1[mask1] - flat2[mask2])

# Calculate the mean difference between the two images
mean_diff = np.mean(diff, axis=0)
print(type(mean_diff))
# Print the mean difference
print("Mean difference between the images:", mean_diff)