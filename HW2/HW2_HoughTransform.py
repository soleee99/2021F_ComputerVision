
import math
import glob
import numpy as np
from PIL import Image


# parameters

datadir = './data'
resultdir='./results'

# TODO:you can calibrate these parameters
sigma=2
highThreshold=130
lowThreshold=20
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
    G = np.flip(G)    # flipping kernel according to def of convolution
    size = G.shape
    kernel_height = size[0] // 2
    kernel_width = size[1] // 2

    Igs = replication_pad(Igs, kernel_height, kernel_width)
    image_height = Igs.shape[0] # padded size
    image_width = Igs.shape[1]
    
    Iconv = np.zeros(original_shape)

    for i in range(kernel_height, image_height - kernel_height):
        for j in range(kernel_width, image_width - kernel_width):
            image_patch = Igs[i-kernel_height:i+kernel_height+1, j-kernel_width:j+kernel_width+1]
            convoluted_value = np.sum(np.multiply(image_patch, G))
            Iconv[i-kernel_height][j-kernel_width] = convoluted_value
    return Iconv


def suppress_result(patch, orientation):
    # orientation is radian direction value for element
    # patch size is 3x3, if <, then suppress
    # return original center value if not need suppressing, else return 0
    # interpolating based on 45 degrees
    values = []
    
    def to_radian(degree):
        # 1 degree = 0.0174533 rad
        return degree * thetaRes

    if (to_radian(22.5) <= orientation and orientation < to_radian(67.5)) \
        or (to_radian(-157.5) <= orientation and orientation < to_radian(-112.5)):
        values.append(patch[0][2])
        values.append(patch[2][0])
    elif (to_radian(67.5) <= orientation and orientation < to_radian(112.5)) \
        or (to_radian(-112.5) <= orientation and orientation < to_radian(-67.5)):
        values.append(patch[0][1])
        values.append(patch[2][1])
    elif (to_radian(112.5) <= orientation and orientation < to_radian(157.5)) \
        or (to_radian(-67.5) <= orientation and orientation < to_radian(-22.5)):
        values.append(patch[0][0])
        values.append(patch[2][2])
    elif (to_radian(-22.5) <= orientation and orientation < to_radian(22.5)) \
        or (to_radian(157.5)<= orientation) \
        or (to_radian(-157.5) > orientation):
        values.append(patch[1][0])
        values.append(patch[1][2])
    
    assert len(values) == 2, "two comparable candidates not chosen"
    # now values holds two interpolated magnitudes
    max_value = max(patch[1][1], max(values))
    
    if max_value > patch[1][1]:
        return 0    # suppressed
    else:
        return patch[1][1]
    


def non_maximum_suppression(mag_image, dir_image):
    """
    Inputs:
        mag_image: magnitude image (Im)
        dir_image: direction image (Io)
    """
    mag_shape = mag_image.shape
    suppressed_image = np.zeros(mag_shape)
    # padded the image all around to make the code easier
    padded_image = replication_pad(mag_image, 1, 1)

    for i in range(1, mag_shape[0]+1):
        for j in range(1, mag_shape[1]+1):
            patch = padded_image[i-1:i+2, j-1:j+2]
            orientation = dir_image[i-1][j-1]
            suppressed_image[i-1][j-1] = suppress_result(patch, orientation)

    return suppressed_image



def double_thresholding(image):
    # 0 for weak, 1 for maybe, 2for definite edges
    
    img_shape = image.shape
    padded_image = replication_pad(image, 1, 1)
    potential_edge = np.zeros(padded_image.shape)

    for i in range(padded_image.shape[0]):
        for j in range(padded_image.shape[1]):
            magnitude = padded_image[i][j]
            if magnitude >= highThreshold:
                potential_edge[i][j] = 2
            elif lowThreshold <= magnitude and magnitude < highThreshold:
                potential_edge[i][j] = 1
            else:
                potential_edge[i][j] = 0


    result_image = image
    for i in range(1, img_shape[0]+1):
        for j in range(1, img_shape[1]+1):
            patch = potential_edge[i-1:i+2, j-1:j+2]
            if patch[1][1] == 1:
                change_to_strong = False
                for m in range(3):
                    for n in range(3):
                        if m == 1 and n == 1:
                            continue
                        if patch[m][n] == 2:
                            # if exits neighboring strong edge, change to strong edge
                            result_image[i-1][j-1] = highThreshold
                            potential_edge[i][j] = 2
                            change_to_strong = True
                            break
                    if change_to_strong:
                        break
                if not change_to_strong:
                    result_image[i-1][j-1] = 0
            elif patch[1][1] == 0:
                result_image[i-1][j-1] = 0

    return result_image




def EdgeDetection(Igs, sigma, highThreshold, lowThreshold):
    # TODO ...
    kernel_size = (5, 5)    # TODO: try changing this
    """
    Inputs:
        Igs: greyscale image
        sigma: stddev to be used for gaussian smoothing before edge detection
            --> find appropriate filter size yourself.
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

    # smooth the original image with gaussian kernel
    kernel_height = kernel_size[0] // 2
    kernel_width = kernel_size[1] // 2

    gk_y = np.fromfunction(lambda x: (np.pi*2*sigma**2)**(-0.5) * np.exp(-0.5*((x-kernel_height)/sigma)**2), (kernel_size[0],))
    y_norm = np.sum(gk_y)
    gk_y = np.expand_dims(gk_y/y_norm, -1)
    gk_x = np.fromfunction(lambda x: (np.pi*2*sigma**2)**(-0.5) * np.exp(-0.5*((x-kernel_width)/sigma)**2), (kernel_size[1],))
    x_norm = np.sum(gk_x)
    gk_x = np.expand_dims(gk_x/x_norm, 0)

    smoothed_image = ConvFilter(Igs, gk_y)
    smoothed_image = ConvFilter(smoothed_image, gk_x)
    # find Ix using sobel filter
    # TODO: maybe change sobel filter size?
    
    sobel_x_3by3 = np.array([[-1,0,1],[-3,0,3],[-1,0,1]])
    #sobel_x_3by3 = np.array([[-1,-2,0,2,1],[-2,-3,0,3,2],[-3,-5,0,5,3],[-2,-3,0,3,2],[-1,-2,0,2,1]])
    Ix = ConvFilter(smoothed_image, sobel_x_3by3)

    # find Iy using sobel filter
    # TODO: maybe change sobel filter size?
    sobel_y_3by3 = np.array([[1,3,1],[0,0,0],[-1,-3,-1]])
    #sobel_y_3by3 = np.array([[1,2,3,2,1],[2,3,5,3,2],[0,0,0,0,0],[-2,-3,-5,-3,-2],[-1,-2,-3,-2,-1]])
    Iy = ConvFilter(smoothed_image, sobel_y_3by3)

    # find Im and Io
    Im = np.sqrt(Ix**2 + Iy**2)
    
    # np.arctan2 returns value btw [-pi, pi]
    Io = np.arctan2(Iy, Ix) # TODO: check for when Ix magnitude is 0

    Im = non_maximum_suppression(Im, Io)
    Image.fromarray(Im).show()
    Im = double_thresholding(Im)
    Image.fromarray(Im).show()
    

    return Im, Io, Ix, Iy

def HoughTransform(Im, rhoRes, thetaRes):
    # TODO ...
    """
    Inputs:
        Im: magnitude image
        rhoRes: rho resolution (unit of voting accumulator)
        thetaRes: theta resolution
    Returns:
        H: Hough transform accumulator
    """
    thetaMax = np.pi * 2
    rhoMax = 700 # maby 800

    thetaSplitValue = 50
    thetaSplitList = np.arange(0, thetaRes, thetaRes/50)
    #print(thetaSplitList)
    thetaList = np.arange(0,thetaMax, thetaRes)
    #print(thetaList)

    H = np.zeros((int(rhoMax / rhoRes), int(thetaMax / thetaRes)))

    for i in range(Im.shape[0]):
        for j in range(Im.shape[1]):
            if Im[i][j] >= highThreshold:
                # if is an edge
                for t, t_val in enumerate(thetaList):
                    thetas = t_val + thetaSplitList
                    rhos = i * np.cos(thetas) + j * np.sin(thetas)
                    rhoMaxInGrid = np.max(rhos)
                    rhoMinInGrid = np.min(rhos)
                    rhoMaxInd = rhoMaxInGrid // rhoRes
                    rhoMinInd = rhoMinInGrid // rhoRes
                    
                    for r in np.arange(rhoMinInd, rhoMaxInd+1):
                        H[int(r)][int(t)] += 1

    Image.fromarray(H).show()
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
        #TODO: why is this needed????" Igs = Igs / 255.
        #Image.fromarray(np.uint8(Igs)).show()  # FIXME:
        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma, highThreshold, lowThreshold)

        #Image.fromarray(Ix).show()  #FIXME:
        #Image.fromarray(Iy).show()  # FIXME:
        
        
        H= HoughTransform(Im, rhoRes, thetaRes)
        """
        lRho,lTheta =HoughLines(H,rhoRes,thetaRes,nLines)
        l = HoughLineSegments(lRho, lTheta, Im)
        """
        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments
    

if __name__ == '__main__':
    main()