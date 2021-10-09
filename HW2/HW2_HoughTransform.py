
import math
import glob
import numpy as np
from PIL import Image


# parameters

datadir = './data'
resultdir='./results'

# you can calibrate these parameters
sigma=2
highThreshold=0.03
lowThreshold=0.01
rhoRes=2
thetaRes=math.pi/180
nLines=20


def replication_pad(image, kernel_height, kernel_width):
    image_height = image.shape[0]
    image_width = image.shape[1]

    out_image = np.zeros((image_height+2*kernel_height, image_width+2*kernel_width))
    out_image[kernel_height:kernel_height+image_height, kernel_width:kernel_width+image_width] = image

    patch = image[:, 0]
    for i in range(kernel_width):
        out_image[kernel_height:kernel_height+image_height,i] = patch
    
    patch = image[:, -1]
    for i in range(kernel_width):
        out_image[kernel_height:kernel_height+image_height,kernel_width+image_width+i] = patch
    
    patch = image[0, :]
    for i in range(kernel_height):
        out_image[i, kernel_width:kernel_width+image_width] = patch
    
    patch = image[-1, :]
    for i in range(kernel_height):
        out_image[kernel_height+image_height+i, kernel_width:kernel_width+image_width] = patch

    out_image[:kernel_height, :kernel_width] = np.full((kernel_height, kernel_width), image[0][0])
    out_image[:kernel_height, kernel_width+image_width:] = np.full((kernel_height, kernel_width), image[0][-1])
    out_image[kernel_height+image_height:, :kernel_width] = np.full((kernel_height, kernel_width), image[-1][0])
    out_image[kernel_height+image_height:, kernel_width+image_width:] = np.full((kernel_height, kernel_width), image[-1][-1])

    return out_image



def ConvFilter(Igs, G):
    # TODO ...
    # under assumption that Igs is already padded, just perform convolution
    original_shape = Igs.shape
    #print(f"original shape: {original_shape}")
    G = np.flip(G)    # flipping kernel according to def of convolution
    size = G.shape
    kernel_height = size[0] // 2
    kernel_width = size[1] // 2

    Igs = replication_pad(Igs, kernel_height, kernel_width)
    #print(f"padded shape: {Igs.shape}")
    image_height = Igs.shape[0] # padded size
    image_width = Igs.shape[1]
    
    Iconv = np.zeros(original_shape)

    for i in range(kernel_height, image_height - kernel_height):
        for j in range(kernel_width, image_width - kernel_width):
            image_patch = Igs[i-kernel_height:i+kernel_height+1, j-kernel_width:j+kernel_width+1]
            convoluted_value = np.sum(np.multiply(image_patch, G))
            Iconv[i-kernel_height][j-kernel_width] = convoluted_value
    #print(f"returning image shape: {Iconv.shape}")
    return Iconv


def EdgeDetection(Igs, sigma, highThreshold, lowThreshold):
    # TODO ...
    """
    Inputs:
        Igs: greyscale image
        sigma: stddev to be used for gaussian smoothing before edge detection
    Returns:
        Im: edge magnitude image
        Io: edge orientation image
        Ix, Iy: edge filter responses in x, and y directions
    
    Process:
        - use ConvFilter function to smooth image with some gaussian kernel
        - to find Ix, convolve smoothed image with x sobel filter
        - to find Iy, convolve smoothed image with y sobel filter
        - then, Im and Io can be calculated using Ix and Iy

        - to have thin edges, make edge detection a 'non-maximal suppression'
            - look at two neighboring pixels in gradient direction,
            - if either of the two has a larger gradient magnitude, 
            - set the edge magnitude at the center pixel to zero
            - TODO: writeup file: explanation and example of non-maximal suppression
        - Double Thresholding
            - magnitude >= highThreshold : strong edge pixel 
                ---> only this should be involved in final image
            - lowThreshold < magnitude < highThreshold : weak edge pixel 
                ---> pick this too if at least one of neighboring pixels are strong edge pixel
            - magnitude <= lowThreshold : suppressed
            - TODO: writeup file: explain how things change when thresholds change

    """

    return Im, Io, Ix, Iy

def HoughTransform(Im, rhoRes, thetaRes):
    # TODO ...


    return H

def HoughLines(H,rhoRes,thetaRes,nLines):
    # TODO ...


    return lRho,lTheta

def HoughLineSegments(lRho, lTheta, Im):
    # TODO ...


    return l

def main():
    """
    x = np.array([[1,2,3],[4,5,6]])
    G = np.ones((3,5)) / 15
    _ = ConvFilter(x, G)
    """
    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        img = Image.open(img_path).convert("L")

        Igs = np.array(img)
        Igs = Igs / 255.

        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma, highThreshold, lowThreshold)
        H= HoughTransform(Im, rhoRes, thetaRes)
        lRho,lTheta =HoughLines(H,rhoRes,thetaRes,nLines)
        l = HoughLineSegments(lRho, lTheta, Im)

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments
    

if __name__ == '__main__':
    main()