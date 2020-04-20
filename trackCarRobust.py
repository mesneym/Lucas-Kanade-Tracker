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


def jacobian(pt):
    dW = np.array([[pt[0], 0, pt[1], 0, 1, 0],
                   [0, pt[0], 0, pt[1], 0, 1]])
    return dW


def affineMatrix(params):
    M = np.array([[1 + params[0], params[2], params[4]],
                  [params[1], 1 + params[3], params[5]]])
    return M


def extractWarpedROI(img, p_prev, rect):
    M = affineMatrix(p_prev)
    I = cv2.warpAffine(img, M, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)  # warped Image
    I = I[rect[0, 1]:rect[1, 1], rect[0, 0]:rect[1, 0]]  # selecting region of interest of warped image
    return I


# def Correction(T,img):
# Tmean = np.mean(T)
# iMean = np.mean(img)

# if(abs(Tmean - imean)< 2)):
# return img
# elif((Tmean-imean)<10):

def zScore(Tmean, img):
    Tmeanmat = np.full(img.shape, Tmean)
    img_mean_matrix = np.full(img.shape, np.mean(img))
    std_ = np.std(img)
    z_score = np.true_divide((img.astype(int) - Tmeanmat.astype(int)), std_)
    dmean = np.mean(img) - Tmean

    if dmean < 10:
        imgshift = -(z_score * std_).astype(int) + img_mean_matrix.astype(int)

    else:
        imgshift = (z_score * std_).astype(int) + img_mean_matrix.astype(int)
        
    return imgshift.astype(dtype=np.uint8)


def gammaCorrection(frame, gamma=1.0):
    newPixel = np.zeros(256, np.uint8)
    for i in range(256):
        newPixel[i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    frame = newPixel[frame]
    return frame


def affineLKtracker(T, img, rect, p_prev):
    oIx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    oIy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    for i in range(1000):
        H = np.zeros((6, 6))
        I = extractWarpedROI(img, p_prev, rect)  # Warped Image ROI
        # I = gammaCorrection(I, 2)  # Correcting image
        I = zScore(np.mean(T), I)
        Ix = extractWarpedROI(oIx, p_prev, rect)  # Warped gradient,Ix ROI
        Iy = extractWarpedROI(oIy, p_prev, rect)  # Warped gradient,Iy ROI

        # uncomment to use double for loops
        # R = np.zeros((6,1))
        # error = (T.astype(int)-I.astype(int))                #computing T(x)- I(w(x,p))

        # for j in range(T.shape[0]):
        # for k in range(T.shape[1]):
        # gradient = np.array([Ix[j,k],Iy[j,k]]).reshape(1,2) #compute warped gradient
        # dW = jacobian([j,k])                                #compute jacobian
        # gradientDw = np.dot(gradient,dW)                    #compute steepest descent,D
        # R += np.dot(gradientDw.T,error[j,k])                #compute transpose(D).(T(x)-I(w(x,p))),R
        # H += np.dot(gradientDw.T,gradientDw)                #compute hessian matrix
        # dp = np.dot(np.linalg.inv(H),R)                             #get change in p

        # uncomment to use meshgrid
        error = (T.astype(int) - I.astype(int)).reshape(-1, 1)  # computing T(x)- I(w(x,p))
        R = np.zeros((T.shape[0] * T.shape[1], 6))
        x, y = np.meshgrid(range(T.shape[1]), range(T.shape[0]))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        for i in range(0, len(x)):
            dW = jacobian([x[i][0], y[i][0]])
            gradient = np.array([Ix[y[i][0]][x[i][0]], Iy[y[i][0]][x[i][0]]])
            R[i] = np.dot(gradient, dW).reshape(1, -1)

        H = R.T @ R
        dp = np.linalg.inv(H) @ R.T @ error
        # ----

        p_prev = p_prev.reshape(6, 1)  # change p_prev to a vector
        p_prev += 100 * dp  # update change in p_prev
        p_prev = p_prev.reshape(6, )  # convert p_prev back to array

        if np.linalg.norm(dp) <= 0.8:
            return p_prev
    return p_prev


def main():
    path = "./Data/Car4/img/*.jpg"
    images, cimages = readImages(path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('trackcarrobust.avi', fourcc, 5.0, (images[0].shape[1], images[0].shape[0]))
    rect_roi = np.array([(69, 52), (175, 136)])
    template = images[0][rect_roi[0][1]:rect_roi[1][1], rect_roi[0][0]:rect_roi[1][0]]
    p_prev = np.zeros(6)
    for i in range(1, len(images)):
        It = images[i]
        p_prev = affineLKtracker(template, It, rect_roi, p_prev)

        M = np.vstack((affineMatrix(p_prev), [0, 0, 1]))  # get new rect coordinates
        x1, y1 = M.dot(np.append(rect_roi[0], 1))[0:2]
        x2, y2 = M.dot(np.append(rect_roi[1], 1))[0:2]

        img = cv2.rectangle(cimages[i], (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.imshow('image1', img)
        out.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
