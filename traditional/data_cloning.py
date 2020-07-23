# Standard imports
import cv2
import numpy as np 
import matplotlib.pyplot as plt 

# Read images
src = cv2.imread("airplane.jpg")
dst = cv2.imread("sky.jpg")
 
print(src.shape)
 
# Create a rough mask around the airplane.
src_mask = np.zeros(src.shape, src.dtype)

plt.figure()
plt.imshow(src_mask)

poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
cv2.fillPoly(src_mask, [poly], (255, 255, 255))

plt.figure()
plt.imshow(src_mask) 
print(src_mask.shape)

# This is where the CENTER of the airplane will be placed
center = (300,300)
 
# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
 
# plt.figure()
# plt.imshow(src)
# plt.figure()
# plt.imshow(output)
plt.show()
# Save result
cv2.imwrite("cloning.jpg", output)