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
    # p2 -> p1 transformation
    # p1 = H p2
    
    N = p1.shape[0]
    A = np.zeros((2*N, 9))
    for i in range(N):
        p1_x = p1[i][0]
        p1_y = p1[i][1]
        p2_x = p2[i][0]
        p2_y = p2[i][1]
        
        A[2*i][0] = p2_x
        A[2*i][1] = p2_y
        A[2*i][2] = 1
        A[2*i][6] = -1 * p1_x * p2_x
        A[2*i][7] = -1 * p1_x * p2_y
        A[2*i][8] = -1 * p1_x

        A[2*i+1][3] = p2_x
        A[2*i+1][4] = p2_y
        A[2*i+1][5] = 1
        A[2*i+1][6] = -1 * p1_y * p2_x
        A[2*i+1][7] = -1 * p1_y * p2_y
        A[2*i+1][8] = -1 * p1_y


    U, s, V = np.linalg.svd(A, full_matrices = True)
    
    # normalizing to return ||h|| = 1
    h = V.T[:,np.argmin(s)]
    h_magnitude = np.sqrt(np.sum(h**2))
    h = h / h_magnitude 
    H = h.reshape(3,3)
    
    return H
    


def compute_h_norm(p1, p2):
    """
    normalize the coordinates, and call compute_h on the normalized coordinates
    normalize x and y coordinates between [0, 1] by dividing by X, and Y respectively.
    """

    norm_mat = np.array([[1/X, 0, 0], [0, 1/Y, 0], [0, 0, 1]])
    p1_norm = np.zeros(p1.shape)
    p2_norm = np.zeros(p2.shape)
    N = p1.shape[0]

    # multiply homogeneous coordinates with normalization matrix
    for i in range(N):
        p1_homo_coor = np.expand_dims(np.concatenate((p1[i], np.array([1])), axis=-1), axis=-1)
        p1_norm[i] = np.squeeze(np.matmul(norm_mat, p1_homo_coor), axis=-1)[:-1]
        p2_homo_coor = np.expand_dims(np.concatenate((p2[i], np.array([1])), axis=-1), axis=-1)
        p2_norm[i] = np.squeeze(np.matmul(norm_mat, p2_homo_coor), axis=-1)[:2]
    
    H = compute_h(p1_norm, p2_norm)

    # undo normalzation
    norm_mat_inv = np.linalg.inv(norm_mat)
    H = np.matmul(np.matmul(norm_mat_inv, H), norm_mat)
    
    """
    # homography accuracy testing code for correspondence points
    for i in range(p1.shape[0]):
        H_inv = np.linalg.inv(H)
        r = np.expand_dims(np.concatenate((p1[i], np.array([1])), axis=-1), axis=-1)
        homo = np.matmul(H_inv, r)
        #print(homo)
        print(f"actual: {p2[i]}, recovered: ({homo[0] / homo[2]}, {homo[1] / homo[2]}")
    """

    return H



def interpolation(img, coordinate):
    """
    img is 2D numpy array (for a single channel)
    interpolated by grabbing the nearest pixel
    returns 0 if out of range
    """
    image_coordinate = np.zeros(2)
    image_coordinate[0] = (Y-1) - (coordinate[1] / coordinate[2])
    image_coordinate[1] = coordinate[0] / coordinate[2]

    # round to nearest integer
    image_coordinate = np.rint(image_coordinate)

    # handle corner cases
    if image_coordinate[0] >= Y or image_coordinate[0] < 0:
        return 0
    elif image_coordinate[1] >= X or image_coordinate[1] < 0:
        return 0
    else:
        return img[int(image_coordinate[0])][int(image_coordinate[1])]



def warp_image(igs_in, igs_ref, H):
    # currently, H changes igs_in -> igs_ref
    merge_y = 2400
    add_y = (merge_y - Y) // 2
    merge_x = 3400
    add_x = merge_x - X

    igs_merge = np.zeros((merge_y, merge_x, 3))
    igs_merge[add_y:add_y+Y, -1*X:, :] = igs_ref

    H_inv = np.linalg.inv(H)
    igs_warp = np.zeros(igs_ref.shape)

    in_channels = [igs_in[:,:,0], igs_in[:,:,1], igs_in[:,:,2]]
    ref_channels = [igs_ref[:,:,0], igs_ref[:,:,1], igs_ref[:,:,2]]
    
    for j in range(X):
        for i in range(Y):
            ref_coordinate = np.array([j, (Y-1)-i, 1])  
            in_coordinate = np.matmul(H_inv, ref_coordinate)    
            igs_warp[i][j][0] = interpolation(in_channels[0], in_coordinate)
            igs_warp[i][j][1] = interpolation(in_channels[1], in_coordinate)
            igs_warp[i][j][2] = interpolation(in_channels[2], in_coordinate)
            
    Image.fromarray(np.uint8(igs_warp)).show()
    
    for j in range(merge_x):
        for i in range(merge_y):
            x_coor = j - add_x
            y_coor = (merge_y-1)-i - add_y
            ref_coordinate = np.array([x_coor, y_coor, 1])
            in_coordinate = np.matmul(H_inv, ref_coordinate) 
            inter_c0 = interpolation(in_channels[0], in_coordinate)
            inter_c1 = interpolation(in_channels[1], in_coordinate)
            inter_c2 = interpolation(in_channels[2], in_coordinate)

            if (j >= add_x) and (add_y <= i and i < add_y+Y):
                if inter_c0 == 0 and inter_c1 == 0 and inter_c2 == 0:
                    # when non overlapping porto2 part, keep the original value
                    inter_c0 = igs_merge[i][j][0]
                    inter_c1 = igs_merge[i][j][1]
                    inter_c2 = igs_merge[i][j][2]
            
            igs_merge[i][j][0] = inter_c0
            igs_merge[i][j][1] = inter_c1
            igs_merge[i][j][2] = inter_c2
               
    Image.fromarray(np.uint8(igs_merge)).show()
    
    return igs_warp, igs_merge



def rectify(igs, p1, p2):
    # TODO ...
    igs_rec = 0

    return igs_rec


def set_cor_mosaic():
    # TODO ...
    """
    p_in and p_ref are N x 2 matrices (correspond to (x,y) coordinates)
    criterion: selected 20 CORNER points from both images.
    """
    

    p_in = np.array([[1187, 1009],
                     [1282, 693],
                     [1447, 694],
                     [1255, 241],
                     [1117, 438],
                     [1554, 624],
                     [1531, 388],
                     [935, 805]])

    p_ref = np.array([[447, 1007],
                     [538, 696],
                     [682, 686],
                     [510, 252],
                     [375, 435],
                     [765, 622],
                     [747, 410],
                     [170, 824]])


    # for actual coordinate (start from 0)
    p_in = p_in - 1
    p_ref = p_ref - 1

    return p_in, p_ref


def set_cor_rec():
    # TODO ...
    c_in = c_ref = 0
    

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
