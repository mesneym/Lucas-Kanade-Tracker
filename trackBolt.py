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
        points2 = np.float32([[266, 80], [307, 80], [307, 143]])
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
path = "Data/Bolt2/img"

# Determining the initial bounding box
img = cv2.imread("Data/Bolt2/img/0001.jpg", 0)
template = img[80:143, 266:307]
# print(template[template.shape[0],template.shape[1]])
# cv2.imshow("template", template)
# img = cv2.rectangle(img, (266, 80), (307, 143), (255, 0, 0), 2)
# cv2.imshow("image", img)

# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
# plt.show()
# top left = [271, 76]
# top right = [305, 76]
# bottom left = [271, 143]
# bottom right = [305, 143]
images = readImages(path)
transformedImages = affineTransformation(images, template)
# print(len(transformedImages))
imageError = computeError(transformedImages, template)
# print(len(imageError))
