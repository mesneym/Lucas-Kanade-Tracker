import cv2
import os
import glob
from matplotlib import pyplot as plt
import numpy as np


def readImages(path):
    images = []
    imagePath = os.path.join(path, '*g')
    imageFiles = glob.glob(imagePath)
    for i in range(1, len(imageFiles)):
        img = cv2.imread(imageFiles[i],0)
        images.append(img)
    return images


def affineTransformation(images, template):
    transformedImages = []
    for image in images:
        points1 = np.float32([[0, 0], [template.shape[0], 0], [template.shape[0], template.shape[1]]])
        points2 = np.float32([[64, 47], [180, 47], [180, 139]])
        # points2 = np.float32([[0, 0], [image.shape[0] - 1, 0], [image.shape[0] - 1, image.shape[1] - 1]])
        matrix = cv2.getAffineTransform(points2, points1)
        newImage = cv2.warpAffine(image, matrix, (template.shape[1], template.shape[0]))
        # cv2.imshow('old', image)
        # cv2.imshow('TI', newImage)
        # if cv2.waitKey(20) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        transformedImages.append(newImage)
    return transformedImages


def computeError(warppedFrames, template):
    error = []
    for image in warppedFrames:
        imageError = np.subtract(template, image)
        error.append(imageError)
    return error


# parameters
path = "Data/Car4/img"

# Determining the initial bounding box
img = cv2.imread("Data/Car4/img/0001.jpg", 0)
template = img[47:139, 64:180]
# print(template[template.shape[0],template.shape[1]])
# cv2.imshow("template", template)
# img = cv2.rectangle(img, (64, 47), (180, 139), (255, 0, 0), 2)
# plt.imshow(img, cmap='gray')
# cv2.imshow("image", img)

# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
# plt.show()
# top left = [64, 47]
# top right = [180, 47]
# bottom left = [64, 139]
# bottom right = [180, 139]
images = readImages(path)
transformedImages = affineTransformation(images, template)
# print(len(transformedImages))
imageError = computeError(transformedImages, template)
# print(len(imageError))
