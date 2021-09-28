import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def reflect_padding(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of three int [height, width]): filter size (e.g. (3,3))
    Return:
        padded image (numpy array)
    """
    
    kernel_height = size[0] // 2
    kernel_width = size[1] // 2

    channels = [input_image[:,:,0], input_image[:,:,1], input_image[:,:,2]]
    padded_list = []

    def pad(image, height, width):

        image_height = image.shape[0]
        image_width = image.shape[1]
        out_image = np.zeros((image_height+2*height, image_width+2*width))
        out_image[height:height+image_height, width:width+image_width] = image

        patch = image[1:1+height, 1:1+width]
        out_image[0:height, 0:width] = np.flip(patch)

        patch = image[1:1+height, image_width-1-width:image_width-1]
        out_image[0:height, width+image_width:] = np.flip(patch)

        patch = image[image_height-1-height:image_height-1, 1:1+width]
        out_image[height+image_height:, 0:width] = np.flip(patch)

        patch = image[image_height-1-height:image_height-1, image_width-1-width:image_width-1]
        out_image[height+image_height:, width+image_width:] = np.flip(patch)
        
        out_image[0:height, width:width+image_width] = np.flip(image[1:1+height, :], axis=0)
        out_image[height+image_height:, width:width+image_width] = np.flip(image[image_height-1-height:image_height-1, :], axis=0)

        out_image[height:height+image_height, 0:width] = np.flip(image[:, 1:1+width], axis=-1)
        out_image[height:height+image_height, width+image_width:] = np.flip(image[:, image_width-1-width:image_width-1], axis=-1)
        return out_image

    for channel in channels:
        """
        mypad = pad(channel, kernel_height, kernel_width)
        np_pad = np.pad(channel, ((kernel_height, kernel_height), (kernel_width, kernel_width)),mode='reflect')
        if not np.array_equal(mypad, np_pad):
            print("not equal")
        """
        padded_list.append(pad(channel, kernel_height, kernel_width))
       
    padded_image = np.stack(padded_list, axis=-1)

    return padded_image


def convolve(input_image, Kernel):
    """
    Args:
        input_image (numpy array): input array
        Kernel (numpy array): kernel shape of (height, width)
    Return:
        convolved image (numpy array)
    """

    # Your code
    # Note that the dimension of the input_image and the Kernel are different.
    # shape of input_image: (height, width, channel)
    # shape of Kernel: (height, width)
    # Make sure that the same Kernel be applied to each of the channels of the input_image
    
    original_shape = input_image.shape
    Kernel = np.flip(Kernel)    # flipping kernel according to def of convolution
    input_image = reflect_padding(input_image, Kernel.shape)
    
    channels = [input_image[:,:,0], input_image[:,:,1], input_image[:,:,2]]
    image_height = input_image.shape[0]
    image_width = input_image.shape[1]

    size = Kernel.shape
    kernel_height = size[0] // 2
    kernel_width = size[1] // 2
    
    output_image = np.zeros(original_shape)

    for c, channel in enumerate(channels):
        for i in range(kernel_height, image_height - kernel_height):
            for j in range(kernel_width, image_width - kernel_width):
                image_patch = channel[i-kernel_height:i+kernel_height+1, j-kernel_width:j+kernel_width+1]
                convoluted_value = np.sum(np.multiply(image_patch, Kernel))
                output_image[i-kernel_height][j-kernel_width][c] = convoluted_value

    return output_image


def median_filter(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,3))
    Return:
        Median filtered image (numpy array)
    """
    for s in size:
        if s % 2 == 0:
            raise Exception("size must be odd for median filter")
    
    # Your code
    output_image = np.zeros((input_image.shape))

    input_image = reflect_padding(input_image, size)
    channels = [input_image[:,:,0], input_image[:,:,1], input_image[:,:,2]]
    image_height = input_image.shape[0]
    image_width = input_image.shape[1]

    kernel_height = size[0] // 2
    kernel_width = size[1] // 2


    for c, channel in enumerate(channels):
        for i in range(kernel_height, image_height - kernel_height):
            for j in range(kernel_width, image_width - kernel_width):
                image_patch = channel[i-kernel_height:i+kernel_height+1, j-kernel_width:j+kernel_width+1]
                median_value = np.median(image_patch)
                output_image[i-kernel_height][j-kernel_width][c] = median_value

    return output_image


def gaussian_filter(input_image, size, sigmax, sigmay):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,.3))
        sigmax (float): standard deviation in X direction
        sigmay (float): standard deviation in Y direction
    Return:
        Gaussian filtered image (numpy array)
    """
    # Your code
    
    # Make kernel
    kernel_height = size[0] // 2
    kernel_width = size[1] // 2

    gk_y = np.fromfunction(lambda x: (np.pi*2*sigmay**2)**(-0.5) * np.exp(-0.5*((x-kernel_height)/sigmay)**2), (size[0],))
    y_norm = np.sum(gk_y)
    gk_y = np.expand_dims(gk_y/y_norm, -1)
    gk_x = np.fromfunction(lambda x: (np.pi*2*sigmax**2)**(-0.5) * np.exp(-0.5*((x-kernel_width)/sigmax)**2), (size[1],))
    x_norm = np.sum(gk_x)
    gk_x = np.expand_dims(gk_x/x_norm, 0)

    
    channels = [input_image[:,:,0], input_image[:,:,1], input_image[:,:,2]]

    
    output1_image = convolve(input_image, gk_y)
    output2_image = convolve(output1_image, gk_x)
    
    return output2_image


if __name__ == '__main__':
    
    #image = np.asarray(Image.open(os.path.join('images', 'baboon.jpeg')).convert('RGB'))
    #image = np.asarray(Image.open(os.path.join('images', 'gaussian_noise.jpeg')).convert('RGB'))
    image = np.asarray(Image.open(os.path.join('images', 'salt_and_pepper_noise.jpeg')).convert('RGB'))
    
    logdir = os.path.join('results', 'HW1_1')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    kernel_1 = np.ones((7,7)) / 49
    sigmax, sigmay = 5, 5
    
    ret = reflect_padding(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'reflect.jpeg'))
        plt.show()

    
    ret = convolve(image.copy(), kernel_1)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'convolve.jpeg'))
        plt.show()
    

    ret = median_filter(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'median.jpeg'))
        plt.show()
    
    ret = gaussian_filter(image.copy(), kernel_1.shape, sigmax, sigmay)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian.jpeg'))
        plt.show()


