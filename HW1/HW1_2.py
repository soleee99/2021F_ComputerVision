import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import utils

def gaussian_pyramid(input_image, level):
    """
    Args:
        input_image (numpy array): input array
        level (int): level of pyramid

    Return:
        Gaussian pyramid (list of numpy array)
    """

    # Your code
    # Note that elements in the list must be arranged in descending order in image resolution (from big image to small image).
    # should have the biggest image at index 0!!
    # cv2's pyrDown function takes care of gaussian filtering before downsizing
    # original image is level 0, returning list shoudl be n+1 length

    gaussian_pyr = [input_image]

    for i in range(level):
        gaussian_pyr.append(utils.down_sampling(gaussian_pyr[i]))

    return gaussian_pyr


def laplacian_pyramid(gaussian_pyr):
    """
    Args:
        gaussian_pyramid (list of numpy array): result from the gaussian_pyramid function

    Return:
        laplacian pyramid (list of numpy array)
    """

    # Your code
    # Note that elements in the list must be arranged in descending order in image resolution (from big image to small image).

    upsampled_pyr = []
    for i in range(1, len(gaussian_pyr)):
        upsampled_pyr.append(utils.up_sampling(gaussian_pyr[i]))
    
    
    laplacian_pyr = []
    for i in range(len(upsampled_pyr)):
        laplacian_pyr.append(utils.safe_subtract(gaussian_pyr[i], upsampled_pyr[i]))
    laplacian_pyr.append(gaussian_pyr[-1])

    return laplacian_pyr


def blend_images(image1, image2, mask, level):
    """
    Args:
        image1 (numpy array): background image
        image2 (numpy array): object image
        mask (numpy array): mask
        level (int): level of pyramid
    Return:
        blended image (numpy array)
    """
    # Your code
    image1_gp = gaussian_pyramid(image1, level)
    image1_lp = laplacian_pyramid(image1_gp)

    image2_gp = gaussian_pyramid(image2, level)
    image2_lp = laplacian_pyramid(image2_gp)

    mask_gp = gaussian_pyramid(mask, level)

    combined_pyr = []
    for i in range(level+1):
        m = mask_gp[i]/255
        combined_img = utils.safe_add((1-m) * image1_lp[i], m * image2_lp[i])
        combined_pyr.append(np.uint8(combined_img))

    combined_pyr.reverse()  # first index has smallest img

    for i in range(len(combined_pyr) - 1):
        upsampled_img = utils.up_sampling(combined_pyr[i])
        combined_pyr[i+1] = utils.safe_add(combined_pyr[i+1], upsampled_img)

    return  combined_pyr[-1]


if __name__ == '__main__':
    hand = np.asarray(Image.open(os.path.join('images', 'hand.jpeg')).convert('RGB'))
    flame = np.asarray(Image.open(os.path.join('images', 'flame.jpeg')).convert('RGB'))
    mask = np.asarray(Image.open(os.path.join('images', 'mask.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_2')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    level = 3

    plt.figure()
    plt.imshow(Image.open(os.path.join('images', 'direct_concat.jpeg')))
    plt.axis('off')
    plt.savefig(os.path.join(logdir, 'direct.jpeg'))
    plt.show()

    ret = gaussian_pyramid(hand, level)
    if ret is not None:
        plt.figure()
        for i in range(len(ret)):
            plt.subplot(1, len(ret), i + 1)
            plt.imshow(ret[i].astype(np.uint8))
            plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian_pyramid.jpeg'))
        plt.show()

        ret = laplacian_pyramid(ret)
        if ret is not None:
            plt.figure()
            for i in range(len(ret)):
                plt.subplot(1, len(ret), i + 1)
                plt.imshow(ret[i].astype(np.uint8))
                plt.axis('off')
            plt.savefig(os.path.join(logdir, 'laplacian_pyramid.jpeg'))
            plt.show()

    ret = blend_images(hand, flame, mask, level)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'blended.jpeg'))
        plt.show()
