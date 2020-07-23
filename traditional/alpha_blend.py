import cv2
import matplotlib.pyplot as plt 
from skimage.io import imread, imsave

# Read the images
# foreground = cv2.imread("orig.png")
foreground = imread("orig.png")
background = imread("bg.png")
alpha = imread("mask.png")
# background = cv2.imread("bg.png")
# alpha = cv2.imread("mask.png")
 
# Convert uint8 to float
foreground = foreground.astype(float)
background = background.astype(float)
 
# Normalize the alpha mask to keep intensity between 0 and 1
alpha = alpha.astype(float)/255
 
# Multiply the foreground with the alpha matte
foreground = cv2.multiply(alpha, foreground)
# plt.figure()
# plt.imshow(foreground)

# Multiply the background with ( 1 - alpha )
background = cv2.multiply(1.0 - alpha, background)
# plt.figure()
# plt.imshow(background) 

# Add the masked foreground and background.
outImage = cv2.add(foreground, background)

print(outImage.min())
# plt.figure()
# plt.imshow(outImage/255)

# cv2.imwrite("output.png", outImage)
imsave("output.png", outImage)
# Display image
# cv2.imshow("outImg", outImage/255)
# cv2.waitKey(0)
plt.show()