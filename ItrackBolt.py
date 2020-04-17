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

def affineMatrix(params):
    W = np.array([[1+params[0,0], params[2,0], params[4,0]],
                  [params[1,0], 1+params[3,0], params[5,0]]])
    # W = np.array([[1+params[0,0], 0, params[4,0]],
                  # [0, 1+params[3,0], params[5,0]]])
    return W 

def extractWarpedROI(img,p_prev,rect):
    M = affineMatrix(p_prev)
    I = cv2.warpAffine(img,M, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP) #warped Image
    I = I[rect[0,1]:rect[1,1],rect[0,0]:rect[1,0]]   # selecting region of interest of warped image
    return I


def affineLKtracker(template,img,rect,p_prev):
    T = template[rect[0,1]:rect[1,1],rect[0,0]:rect[1,0]]   # selecting region of interest
    oIx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    oIy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
     
    for i in range(100):
        R = np.zeros((6,1))
        H = np.zeros((6,6))                  
        I = extractWarpedROI(img,p_prev,rect) #Warped Image ROI
        Ix= extractWarpedROI(oIx,p_prev,rect)  #Warped gradient,Ix ROI
        Iy= extractWarpedROI(oIy,p_prev,rect)  #Warped gradient,Iy ROI
        error = T-I                           #computing T(x)- I(w(x,p))
        
        for j in range(T.shape[0]):
            for k in range(T.shape[1]):
                # print(Ix.shape)
                # print(T.shape)
                # print(i)
                # print("========")
                # print(" ")
                gradient = np.array([Ix[j,k],Iy[j,k]]).reshape(1,2) #compute warped gradient
                dW = jacobian([j,k])                                #compute jacobian
                gradientDw = np.dot(gradient,dW)                    #compute steepest descent,D
                R += np.dot(gradientDw.T,error[j,k])                #compute transpose(D).(T(x)-I(w(x,p)))
                H += np.dot(gradientDw.T,gradientDw)                #compute hessian matrix  

        dp = np.dot(np.linalg.inv(H),R)                             #get change in p
        p_prev += 0.01*dp                                           #update change in p

        if(np.linalg.norm(dp)<= 0.01):
            return p_prev
            # print(p_prev)
    return p_prev



def main():
    path = "./Data/Bolt2/img/*.jpg"
    images = readImages(path)
    rect_roi = np.array([(266, 80), (307, 143)])
    p_prev = np.array([0.0,0.0,0.0,0.0,0.0,0.0]).reshape(6,1)
    for i in range(len(images)-1):
        It0 = images[i]
        It1 = images[i+1]
        # It0 = cv2.addWeighted(It0, 1, It0,0,10)
        # It1 = cv2.addWeighted(It1, 1, It1,0,10)
        img1 = cv2.rectangle(It0, tuple(rect_roi[0]), tuple(rect_roi[1]), (255, 0, 0), 2)
        p_prev = affineLKtracker(It0,It1,rect_roi,p_prev)
        
        M = np.vstack((affineMatrix(p_prev),[0,0,1]))       #get new rect coordinates
        rect_roi[0] = M.dot(np.append(rect_roi[0],1))[0:2]
        rect_roi[1] = M.dot(np.append(rect_roi[1],1))[0:2]

        img2 = cv2.rectangle(It1, tuple(rect_roi[0]), tuple(rect_roi[1]), (255, 0, 0), 2)

        cv2.imshow('image1',img1)
        cv2.imshow('image2',img2)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

if __name__=="__main__":
    main()





