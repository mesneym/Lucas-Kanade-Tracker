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

def derivativeaffineWarp(pt):
    dW = np.array([[pt[0], 0, pt[1], 0, 1, 0],
                   [0, pt[0], 0, pt[1], 0, 1]])
    return dW

def affineWarp(pt,params):
    p = np.append(pt,1)
    W = np.array([[1+params[0], params[2], params[4]],
                  [params[1], 1+params[3], params[5]]])
    return np.dot(W,p)

def warpROI(I,rect,params):
    minX = np.min(rect[:,0])
    minY = np.min(rect[:,1])
    maxX = np.max(rect[:,0])
    maxY = np.max(rect[:,1])
    
    wI = np.zeros((I.shape),dtype=np.uint8)
    for i in range(minY,maxY):
        for j in range(minX,maxX):
            # print(i,j)
            # print(affineWarp([i,j],params))
            # print("===========")
            # print('')
            x,y = affineWarp([i,j],params)
            wI[i,j] = I[x,y]
    return wI 

def affineLKtracker(T,I,rect,p_prev):
    wI = warpROI(T,rect,p_prev)  

    # difference between image
    error = np.subtract(T,wI)
    
    #compute gradient
    Ix = cv2.Sobel(wI,cv2.CV_64F,1,0,ksize=5)
    Iy = cv2.Sobel(wI,cv2.CV_64F,0,1,ksize=5)
    gradientI = np.column_stack((Ix,Iy))
   







    # img = cv2.rectangle(T, (266, 80), (307, 143), (255, 0, 0), 2)
    # cv2.imshow("image", img)
    # cv2.imshow("image2",wI)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
            # cv2.destroyAllWindows()
  
    

def main():
    path = "Data/Bolt2/img"
    images = readImages(path)
   
    It0 = cv2.imread("Data/Bolt2/img/0001.jpg", 0)
    It1 = cv2.imread("Data/Bolt2/img/0002.jpg",0)

    rect_roi = np.array([(266, 80), (307, 143)])
    p_initial = [0,0,0,0,0,0]
    print(affineLKtracker(It0,It1,rect_roi,p_initial))
     

    #template
    # template = img[80:143, 266:307]
    # img = cv2.rectangle(img, (266, 80), (307, 143), (255, 0, 0), 2)
    # cv2.imshow("image", img)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
            # cv2.destroyAllWindows()
    
     
    # transformedImages = affineTransformation(images, template)
    # imageError = computeError(transformedImages, template)


if __name__=="__main__":
    main()






# parameters
# path = "Data/Bolt2/img"

# Determining the initial bounding box
# img = cv2.imread("Data/Bolt2/img/0001.jpg", 0)
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
# images = readImages(path)
# print(len(transformedImages))
# print(len(imageError))


