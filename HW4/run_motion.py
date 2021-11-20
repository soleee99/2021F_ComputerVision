import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold

from PIL import Image   # TODO: delete this

def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and 
    # RectBivariateSpline. Never use OpenCV.
    # Gx, Gy are 2d array

    
    warped_coor_x = np.zeros(img1.shape)
    warped_coor_y = np.zeros(img1.shape)

    img1_y, img1_x = img1.shape
    img2_y, img2_x = img2.shape

    # contruct interpolation for I(t+1)
    interpolated_img2 = RectBivariateSpline(np.linspace(0, img2_y-1, img2_y), 
                                            np.linspace(0, img2_x-1, img2_x), 
                                            img2)

    M = np.array([[p[0]+1, p[2], p[4]],
                  [p[1], p[3]+1, p[5]]])

    # warp img1 (I(t)), store warped coordinates in warped_coor_x and y
    for y in range(img1_y):
        for x in range(img1_x):
            warped_coor = np.matmul(M, np.array([x, y, 1]))
            warped_coor_x[y][x] = warped_coor[0]
            warped_coor_y[y][x] = warped_coor[1]
    
    # I(W(x;p))
    warped_image = interpolated_img2.ev(warped_coor_y, warped_coor_x)
    #Image.fromarray(np.uint8(warped_image)).show()
    # T(x) - I(W(x;p)), T(x) = I(t) = img1 here
    diff = img1 - warped_image
    #Image.fromarray(np.uint8(diff)).show()
    diff = diff.reshape(-1, 1)  # (N, 1)

    Gmax = max(np.max(Gx), np.max(Gy))  
    # now make grad*(jacobian of W WRT p)
    grad_jacobian = np.zeros((img1_x*img1_y, 6))  # (N, 6)
    for y in range(img1_y):
        for x in range(img1_x):
            jacobian = np.array([[x, 0, y, 0, 1, 0],
                                 [0, x, 0, y, 0, 1]])
            grad = np.array([[Gx[y][x], Gy[y][x]]]) / Gmax
            index = y*img1_x + x
            grad_jacobian[index] = np.squeeze(np.matmul(grad, jacobian), axis=0)
    
    grad_jacobian = grad_jacobian * Gmax

    H = np.matmul(grad_jacobian.T, grad_jacobian)   # (6, 6)

    tmp = np.matmul(np.linalg.inv(H), grad_jacobian.T)
    dp = np.squeeze(np.matmul(tmp, diff), axis=-1)


    ### END CODE HERE ###
    return dp

def subtract_dominant_motion(img1, img2):
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = 5) # do not modify this
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = 5) # do not modify this
    print(f"GX size: {Gx.shape}")
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and 
    # RectBivariateSpline. Never use OpenCV.

   
    # initialize p
    dp = np.array([1, 0, 0, 0, 1, 0])
    p = np.zeros(6)    

    dp = lucas_kanade_affine(img1, img2, p, Gx, Gy)
    p += dp


    moving_image = np.abs(img2 - img1) # you should delete this
    
    th_hi = 0.2 * 256 # you can modify this
    th_lo = 0.15 * 256 # you can modify this

    
    ### END CODE HERE ###

    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)
    return hyst

if __name__ == "__main__":
    data_dir = 'data'
    video_path = 'motion.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 150/20, (636, 318))
    tmp_path = os.path.join(data_dir, "organized-{}.jpg".format(0))
    T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
    for i in range(0, 50):
        img_path = os.path.join(data_dir, "organized-{}.jpg".format(i))
        I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        clone = I.copy()
        moving_img = subtract_dominant_motion(T, I)
        clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
        clone[moving_img, 2] = 522
        out.write(clone)
        T = I
    out.release()
    