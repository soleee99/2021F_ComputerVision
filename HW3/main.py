import math
import numpy as np
from PIL import Image

"""
For convenience of matrix multiplication and stuff.....
consider all image coordinates as;
         _______________________________
y = Y   |                               |
        |                               |
        |                               |
        |                               |
        |                               |
y = 0   |_______________________________|
        x = 0                         x = X
"""

# porto image Y and X values
X = 1600
Y = 1200

def compute_h(p1, p2):
    # TODO ...

    return H

def compute_h_norm(p1, p2):
    # TODO ...

    return H

def warp_image(igs_in, igs_ref, H):
    # TODO ...

    return igs_warp, igs_merge

def rectify(igs, p1, p2):
    # TODO ...

    return igs_rec

def set_cor_mosaic():
    # TODO ...
    """
    p_in and p_ref are N x 2 matrices (correspond to (x,y) coordinates)
    criterion: selected 20 CORNER points from both images.
    """
    
    p_in = np.array([[1188, 1012],
                     [1227, 1017],
                     [1261, 944],
                     [1288, 949],
                     [1284, 785],
                     [1442, 793],
                     [1286, 694],
                     [1146, 697],
                     [1241, 659],
                     [1287, 621],
                     [1370, 619],
                     [1462, 620],
                     [1368, 477],
                     [1118, 446],
                     [1167, 410],
                     [1228, 365],
                     [1402, 393],
                     [1334, 288],
                     [1287, 255],
                     [1254, 244]])

    p_ref = np.array([[448, 1006],
                      [480, 1008],
                      [513, 930],
                      [540, 936],
                      [536, 775],
                      [675, 774],
                      [540, 689],
                      [679, 688],
                      [493, 655],
                      [539, 626],
                      [612, 618],
                      [689, 614],
                      [613, 487],
                      [376, 440],
                      [428, 409],
                      [487, 369],
                      [641, 410],
                      [584, 305],
                      [538, 271],
                      [510, 252]])
    
    # for actual coordinate (start from 0)
    p_in = p_in - 1
    p_ref = p_ref - 1

    return p_in, p_ref

def set_cor_rec():
    # TODO ...
    

    return c_in, c_ref


def main():
    ##############
    # step 1: mosaicing
    ##############

    # read images
    img_in = Image.open('data/porto1.png').convert('RGB')
    img_ref = Image.open('data/porto2.png').convert('RGB')

    # shape of igs_in, igs_ref: [y, x, 3]
    # the two images have the same shape
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)
    
    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)
    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save('porto1_warped.png')
    img_merge.save('porto_mergeed.png')

    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open('data/iphone.png').convert('RGB')
    igs_rec = np.array(img_rec)

    c_in, c_ref = set_cor_rec()

    igs_rec = rectify(igs_rec, c_in, c_ref)

    img_rec = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec.save('iphone_rectified.png')

if __name__ == '__main__':
    main()
