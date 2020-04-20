import cv2
import os
import glob
from matplotlib import pyplot as plt
import numpy as np


def readImages(path):
    img_array = []
    imgc = []
    names = []
    for filename in glob.glob(path):
        names.append(filename)
    names.sort()

    for filename in names:
        img = cv2.imread(filename, 0)
        img1 = cv2.imread(filename)
        img_array.append(img)
        imgc.append(img1)
    return img_array, imgc

def derivativeaffineWarp(pt):
    dW = np.array([[pt[0], 0, pt[1], 0, 1, 0],
                   [0, pt[0], 0, pt[1], 0, 1]])
    return dW

def affineWarp(pt,params):
    #params is a column vector
    p = np.append(pt,1)
    W = np.array([[1+params[0,0], params[2,0], params[4,0]],
                  [params[1,0], 1+params[3,0], params[5,0]]])
    return np.dot(W,p).astype(int)

def warpROI(I,rect,params):
    minX = np.min(rect[:,0])
    minY = np.min(rect[:,1])
    maxX = np.max(rect[:,0])
    maxY = np.max(rect[:,1])
    
    wI = np.zeros((I.shape),dtype=np.uint8)
    for i in range(minY-4,maxY+4):
        for j in range(minX-4,maxX+4):
                x,y = affineWarp([i,j],params)
                wI[i,j] = I[x,y]
    return wI 

def affineLKtracker(T,I,rect,p_prev):
    error = np.subtract(T,I)
    Ix = cv2.Sobel(I,cv2.CV_64F,1,0,ksize=7)
    Iy = cv2.Sobel(I,cv2.CV_64F,0,1,ksize=7)

    for i in range(5):

        minX = np.min(rect[:,0])
        minY = np.min(rect[:,1])
        maxX = np.max(rect[:,0])
        maxY = np.max(rect[:,1])

        result = np.zeros((6,1))
        H = np.zeros((6,6))
        for i in range(minY,maxY):
            for j in range(minX,maxX):
                x,y = affineWarp([i,j],p_prev)
                gradient = np.array([Ix[x,y],Iy[x,y]]).reshape(1,2)
                dW = derivativeaffineWarp([i,j])
                gradientDw = np.dot(gradient,dW)
                result += np.dot(gradientDw.T,T[i,j]-I[x,y])
                H += np.dot(gradientDw.T,gradientDw) 

        dp = np.dot(np.linalg.inv(H),result)
        p_prev += 10*dp
        if(np.linalg.norm(dp)<= 0.05):
            return p_prev
    return p_prev


def main():
    path = "./Data/DragonBaby/img/*.jpg"
    images, cimages = readImages(path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('trackbolt2.avi', fourcc, 5.0, (images[0].shape[1], images[0].shape[0]))
    rect_roi = np.array([(156, 68), (214, 146)])
    p_prev = np.array([0.0,0.0,0.0,0.0,0.0,0.0]).reshape(6,1)
    for i in range(len(images)-1):
        It0 = images[i]
        It1 = images[i+1] 
        p_prev = affineLKtracker(It0,It1,rect_roi,p_prev)
        img1 = cv2.rectangle(It0, tuple(rect_roi[0]), tuple(rect_roi[1]), (255, 0, 0), 2)

        rect_roi[0] = affineWarp(rect_roi[0],p_prev)
        rect_roi[1] = affineWarp(rect_roi[1],p_prev)
        img2 = cv2.rectangle(cimages[i], tuple(rect_roi[0]), tuple(rect_roi[1]), (255, 0, 0), 2)

        cv2.imshow('image1',img1)
        cv2.imshow('image2',img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=="__main__":
    main()









