import cv2
import os
import glob
from matplotlib import pyplot as plt
import numpy as np


def readImages(path):
    img_array = []
    names = []
    for filename in glob.glob(path):
        names.append(filename)
    names.sort()

    for filename in names:
        img = cv2.imread(filename,0)
        img_array.append(img)
    return img_array

def jacobian(pt):
    dW = np.array([[pt[0], 0, pt[1], 0, 1, 0],
                   [0, pt[0], 0, pt[1], 0, 1]])
    return dW

def affineWarp(pt,params):
    p = np.append(pt,1)
    W = np.array([[1+params[0,0], params[2,0], params[4,0]],
                  [params[1,0], 1+params[3,0], params[5,0]]])
    # W = np.array([[1+params[0,0], 0, params[4,0]],
                  # [0, 1+params[3,0], params[5,0]]])
    return np.dot(W,p).astype(int)

def affineLKtracker(T,I,rect,p_prev):
    Ix = cv2.Sobel(I,cv2.CV_64F,1,0,ksize=7)
    Iy = cv2.Sobel(I,cv2.CV_64F,0,1,ksize=7)
    minX = np.min(rect[:,0])
    minY = np.min(rect[:,1])
    maxX = np.max(rect[:,0])
    maxY = np.max(rect[:,1])

    for i in range(10):
        result = np.zeros((6,1))
        H = np.zeros((6,6))
        for j in range(minY,maxY):
            for k in range(minX,maxX):
                x,y = affineWarp([j,k],p_prev)                      #warp image points
                error = T[j,k]-I[x,y]                               #error T(x)-I(w(x,p))
                gradient = np.array([Ix[x,y],Iy[x,y]]).reshape(1,2) #compute warped gradient
                dW = jacobian([j,k])                                #compute jacobian
                gradientDw = np.dot(gradient,dW)                    #compute steepest descent,D
                result += np.dot(gradientDw.T,error)                #compute transpose(D).(T(x)-I(w(x,p)))
                H += np.dot(gradientDw.T,gradientDw)                #compute hessian matrix  

        dp = np.dot(np.linalg.inv(H),result)
        p_prev += dp

        if(np.linalg.norm(dp)<= 0.1):
            print(p_prev)
            return p_prev
    return p_prev


def main():
    path = "./Data/Car4/img/*.jpg"
    images = readImages(path)
    rect_roi = np.array([(64, 47), (180, 139)])
    p_prev = np.array([0.0,0.0,0.0,0.0,0.0,0.0]).reshape(6,1)
    for i in range(len(images)-1):
        It0 = images[i]
        It1 = images[i+1] 
        p_prev = affineLKtracker(It0,It1,rect_roi,p_prev)
        img1 = cv2.rectangle(It0, tuple(rect_roi[0]), tuple(rect_roi[1]), (255, 0, 0), 2)

        rect_roi[0] = affineWarp(rect_roi[0],p_prev)
        rect_roi[1] = affineWarp(rect_roi[1],p_prev)
        img2 = cv2.rectangle(It1, tuple(rect_roi[0]), tuple(rect_roi[1]), (255, 0, 0), 2)

        cv2.imshow('image1',img1)
        cv2.imshow('image2',img2)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

if __name__=="__main__":
    main()

