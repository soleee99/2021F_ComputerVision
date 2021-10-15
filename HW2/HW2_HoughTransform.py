
import math
import glob
import numpy as np
from PIL import Image, ImageDraw


# parameters

datadir = './data'
resultdir='./results'

# TODO:you can calibrate these parameters
sigma=2
highThreshold=100
lowThreshold=20
rhoRes=1.5
thetaRes=math.pi/180
nLines=8

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
        return degree * (math.pi/180)

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
    
    sobel_x_3by3 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    #sobel_x_3by3 = np.array([[-1,-2,0,2,1],[-2,-3,0,3,2],[-3,-5,0,5,3],[-2,-3,0,3,2],[-1,-2,0,2,1]])
    Ix = ConvFilter(smoothed_image, sobel_x_3by3)

    # find Iy using sobel filter
    # TODO: maybe change sobel filter size?
    sobel_y_3by3 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    #sobel_y_3by3 = np.array([[1,2,3,2,1],[2,3,5,3,2],[0,0,0,0,0],[-2,-3,-5,-3,-2],[-1,-2,-3,-2,-1]])
    Iy = ConvFilter(smoothed_image, sobel_y_3by3)

    # find Im and Io
    Im = np.sqrt(Ix**2 + Iy**2)
    
    # np.arctan2 returns value btw [-pi, pi]
    Io = np.arctan2(Iy, Ix) # TODO: check for when Ix magnitude is 0

    Im = non_maximum_suppression(Im, Io)
    #Image.fromarray(Im).show()
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
    
    #Image.fromarray(Im).show()

    thetaList = np.arange(-0.5* math.pi, 0.5*math.pi, thetaRes)
    #print(thetaList)
    rhoMax = int(np.ceil(np.sqrt(Im.shape[0]**2 + Im.shape[1]**2)))
    thetaMax = 0.5*math.pi

    """
    H = np.zeros((int(rhoMax / rhoRes) + 1, int(thetaMax / thetaRes) + 1))

    for i in range(Im.shape[0]):
        for j in range(Im.shape[1]):
            if Im[Im.shape[0] - i - 1][j] >= highThreshold:
                # if is an edge
                for theta in thetaList:
                    #thetas = t_val + thetaSplitList
                    rho = j * np.cos(theta) + i * np.sin(theta) 
                    #rhoMaxInGrid = np.max(rhos)
                    #rhoMinInGrid = np.min(rhos)
                    #rhoMaxInd = rhoMaxInGrid // rhoRes
                    #rhoMinInd = rhoMinInGrid // rhoRes
                    if rho < 0:
                        rho = rho * -1
                        theta += math.pi
                        if theta >= 2* math.pi:
                            theta -= 2*math.pi
                        
                    rho_ind = int(rho / rhoRes)
                    theta_ind = int(theta / thetaRes)

                    H[rho_ind][theta_ind] += 1
                    #for r in np.arange(rhoMinInd, rhoMaxInd+1):
                        #print(f"({r}, {t})")
                    #    H[int(r)][int(t)] += 1
    """

    H = np.zeros((int(2*rhoMax/rhoRes) + 1, int(2* thetaMax / thetaRes) + 1))
    for i in range(Im.shape[0]):
        for j in range(Im.shape[1]):
            if Im[Im.shape[0] - i - 1][j] >= highThreshold:
                for theta in thetaList:
                    rho = j * np.cos(theta) + i * np.sin(theta) + rhoMax
                    theta += thetaMax
                    rho_ind = int(rho / rhoRes)
                    theta_ind = int(theta / thetaRes)

                    H[rho_ind][theta_ind] += 1
                    

    #Image.fromarray(H).show()
    
    return H
    
    



def non_maximum_suppression_for_houghlines(patch, patch_size, countThreshold):
    count = patch[patch_size][patch_size]
    
    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            if i == patch_size and j == patch_size:
                continue
            
            nearby_count = patch[i][j]
            if count < nearby_count :
                return 0
    if count >= countThreshold:
        return count
    else:
        return 0



def HoughLines(H,rhoRes,thetaRes,nLines):
    # TODO ...
    """
    Inputs:
        - H: hough transform accumulator
        - rhoRes, thetaRes: resolution, neede to recover actual rho and theta 
                            value from the hough transform accumulator index
        - nLines: number of lines (hopefully different local maximas of H) to return
    Outputs:
        - lRho, lTheta: nLines-sized arrays that hold the line values!
    Tip:
        - maybe set a count threshold to consider local maxima
        - pad the thing, use strict less
        - patch consideration?
    """
    countThreshold = 96
    patch_size = 10 # how many pixels above/below/side

    original_shape = H.shape
    padded_H = replication_pad(H, patch_size, patch_size)
    suppressed_H = np.zeros(original_shape)

    for i in range(patch_size, original_shape[0]+patch_size):
        for j in range(patch_size, original_shape[1]+patch_size):
            patch = padded_H[i-patch_size:i+(patch_size+1), j-patch_size:j+(patch_size+1)]
            suppressed_H[i-patch_size][j-patch_size] = non_maximum_suppression_for_houghlines(patch, patch_size, countThreshold)
    
    

    rhos = []
    thetas = []
    counts = []
    rhoMax = H.shape[0] // 2 * rhoRes
    thetaMax =  0.5 * math.pi
    #Image.fromarray(suppressed_H).show()
    for i in range(original_shape[0]):
        for j in range(original_shape[1]):
            if suppressed_H[i][j] != 0:
                counts.append(suppressed_H[i][j])
                rhos.append(i*rhoRes-rhoMax)
                thetas.append(j*thetaRes - thetaMax)
    
    zipped = list(zip(counts, rhos, thetas))
    
    sorted_result = sorted(zipped, key=lambda x : x[0]) # sort using number of counts
    #print(sorted_result)
    sorted_result = sorted_result[-nLines:]
    
    lRho = np.array([rho for c, rho, theta in sorted_result]) #* rhoRes
    lTheta = np.array([theta for c, rho, theta in sorted_result]) #* thetaRes
    #print(list(zip(lRho, lTheta)))
    #print("\n")
    return lRho,lTheta


def HoughTransformKeepPoints(Im, rhoRes, thetaRes):
    thetaList = np.arange(-0.5* math.pi, 0.5*math.pi, thetaRes)
    rhoMax = int(np.ceil(np.sqrt(Im.shape[0]**2 + Im.shape[1]**2)))
    thetaMax = 0.5*math.pi

    # make this to keep in track of what points voted for a line
    points = []
    for i in range(int(2*rhoMax/rhoRes) + 1):
        for_row = []
        for j in range(int(2* thetaMax / thetaRes) + 1):
            for_row.append([])
        points.append(for_row)

        
    for i in range(Im.shape[0]):
        for j in range(Im.shape[1]):
            if Im[Im.shape[0] - i - 1][j] >= highThreshold:
                for theta in thetaList:
                    rho = j * np.cos(theta) + i * np.sin(theta) + rhoMax
                    theta += thetaMax
                    rho_ind = int(rho / rhoRes)
                    theta_ind = int(theta / thetaRes)

                    points[rho_ind][theta_ind].append((Im.shape[0] - i - 1,j))
    
    

    return points


def get_start_end(points, segment_threshold, r, t):
    case = 0
    if np.cos(t) == 0.0:
        case = 1
    elif np.sin(t) == 0.0:
       case = 2
    elif r / np.cos(t) == float("inf") or r / np.cos(t) == float("-inf"):
        case = 1
    elif r / np.sin(t) == float("inf") or r / np.sin(t) == float("-inf"):
        case = 2
    else:
        case = 3

    #print(f"\nsegment threshold: {segment_threshold}, case: {case}")
    #print(f"entire sorted list: {points}")
    groups = []
    ind = 0
    x = points[0][0]
    y = points[0][1]
    if case == 3:
        slope = -1 * (np.cos(t) / np.sin(t))
        if abs(slope) > 25:
            case = 2
        if segment_threshold == 3:
            min_slope = 0.02
        elif segment_threshold == 5:
            min_slope = 0.017
        else:
            min_slope = 0.017
        if abs(slope) <= min_slope:
            case = 1
        #print(f"case 3 slope: {slope}, changed to {case}")
        
    groups_added = False
    for i, point in enumerate(points):
        if ((case == 1 and (abs(point[1] - y) > segment_threshold)) or \
            (case == 2 and (abs(point[0] - x) > segment_threshold)) or \
            (case == 3 and (abs(point[0] - x) > segment_threshold and abs(point[1] - y) > segment_threshold))):
            
            #print(f"add groups:{points[ind:i]}, from {ind} to {i}")
            groups.append(points[ind:i])
            ind = i
            groups_added = True
        x = point[0]
        y = point[1]
    
    if groups_added:
        #print(f"add groups:{points[ind:]}, from {ind} to end")
        groups.append(points[ind:])
    
    if len(groups) == 0:
        groups = [points]
    
    
    max_len = 0
    max_group = groups[0]
    for g in groups:
        
        if len(g) > max_len:
            max_len = len(g)
            max_group = g
    
    return max_group[0], max_group[-1]


def HoughLineSegments(lRho, lTheta, Im, segment_threshold):
    # TODO ...
    """
    If the slope and gradient direction for that point are orthogonal....
    """
    l = []

    H = HoughTransform(Im, rhoRes, thetaRes)
    countThreshold = 96
    patch_size = 10 # how many pixels above/below/side

    original_shape = H.shape
    padded_H = replication_pad(H, patch_size, patch_size)
    suppressed_H = np.zeros(original_shape)

    for i in range(patch_size, original_shape[0]+patch_size):
        for j in range(patch_size, original_shape[1]+patch_size):
            patch = padded_H[i-patch_size:i+(patch_size+1), j-patch_size:j+(patch_size+1)]
            suppressed_H[i-patch_size][j-patch_size] = non_maximum_suppression_for_houghlines(patch, patch_size, countThreshold)
    
    counts = []
    line_points = []
    for i in range(suppressed_H.shape[0]):
        for j in range(suppressed_H.shape[1]):
            if suppressed_H[i][j]!= 0:
                counts.append(suppressed_H[i][j])
                line_points.append((i,j))
    
    zipped = list(zip(counts, line_points))
    sorted_result = sorted(zipped, key=lambda x : x[0]) # sort using number of counts
    sorted_result = sorted_result[-nLines:]

    points = HoughTransformKeepPoints(Im, rhoRes, thetaRes)
    thetaMax = 0.5 * math.pi
    rhoMax = int(np.ceil(np.sqrt(Im.shape[0]**2 + Im.shape[1]**2)))

    for c, (rho_ind, theta_ind) in sorted_result:
        
        p = points[rho_ind][theta_ind]  # p is list of points that voted for this (rho, theta) line
        sorted_points = sorted(p, key=lambda x: (x[0], x[1]))
        start, end = get_start_end(sorted_points, segment_threshold, rho_ind * rhoRes - rhoMax, theta_ind*thetaRes-thetaMax)
    
        l.append({'start':start, 'end':end})

    return l



def find_xy_tuples(y_shape, x_shape, rho, theta):
    # rho = x cos(theta) + y sin(theta)
    # to plot, (0,0) is top left corner
    """
    coordinates = []
    for r, t in zip(rho, theta):
        x_intersect = int(r / np.cos(t))
        y_intersect = int(r / np.sin(t))
        slope = -1 * (np.cos(t) / np.sin(t))
        if slope > 0:
            if y_intersect >= 0:
                if slope * x_shape + y_intersect <= y_shape:
                    coordinates.append([(y_shape-y_intersect,0), (y_shape-(int(slope * x_shape) + y_intersect), x_shape-1)])
                else:
                    coordinates.append([(y_shape-y_intersect,0), (0, int((1/slope)*y_shape) + x_intersect)])
            else:
                if slope * x_shape + y_intersect <= y_shape:
                    coordinates.append([()])




    x = rho / np.cos(theta)
    y = rho / np.sin(theta)

    coordinates = []
    for i, j in zip(x, y):
        coordinates.append([(i,0),(0,y_shape - j)])
    """
    
    coordinates = []
    for r, t in zip(rho, theta):
        if np.cos(t) == 0.0:
            y_intersect = int(r / np.sin(t))
            coordinates.append([(0, y_shape - (y_intersect)), (x_shape, y_shape - (y_intersect))])
        elif np.sin(t) == 0.0:
            x_intersect = int(r / np.cos(t))
            coordinates.append([(x_intersect, 0), (x_intersect, y_shape)])
        
        elif r / np.cos(t) == float("inf") or r / np.cos(t) == float("-inf"):
            y_intersect = int(r / np.sin(t))
            coordinates.append([(0, y_shape - (y_intersect)), (x_shape, y_shape - (y_intersect))])
        elif r / np.sin(t) == float("inf") or r / np.sin(t) == float("-inf"):
            x_intersect = int(r / np.cos(t))
            coordinates.append([(x_intersect, 0), (x_intersect, y_shape)])
        
        else:
            x_intersect = int(r / np.cos(t))
            y_intersect = int(r / np.sin(t))
            slope = -1 * (np.cos(t) / np.sin(t))
            coordinates.append([(0, y_shape - (y_intersect)), (x_shape, y_shape - int(slope * x_shape + y_intersect))])

    return coordinates



def plot_image(img, coordinates):
    new_img = img.copy()
    draw = ImageDraw.Draw(new_img)
    #print(coordinates)
    for coordinate in coordinates:
        #draw = ImageDraw.Draw(img)
        draw.line(coordinate)
    new_img.show()



def plot_image_dict(img, dict_points):
    new_img = img.copy()
    draw = ImageDraw.Draw(new_img)
    coordinates = []
    for point in dict_points:
        coordinates.append([(point['start'][1], point['start'][0]), (point['end'][1], point['end'][0])])
    #print(coordinates)
    for coordinate in coordinates:
        #draw = ImageDraw.Draw(img)
        draw.line(coordinate)
    new_img.show()



def main():
    
    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        img = Image.open(img_path).convert("L")
        #print(img.size)
        
       

        Igs = np.array(img)
        #TODO: why is this needed????" Igs = Igs / 255.
        #Image.fromarray(np.uint8(Igs)).show()  # FIXME:
        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma, highThreshold, lowThreshold)

        
        H = HoughTransform(Im, rhoRes, thetaRes)
        
        lRho,lTheta = HoughLines(H,rhoRes,thetaRes,nLines)
        
        coordinates = find_xy_tuples(Igs.shape[0], Igs.shape[1], lRho, lTheta)
        plot_image(img, coordinates)
        
        if 'img01' in img_path:
            segment_threshold = 3
        elif 'img02' in img_path:
            segment_threshold = 5
        elif 'img03' in img_path:
            segment_threshold = 800
        elif 'img04' in img_path:
            segment_threshold = 6

        l = HoughLineSegments(lRho, lTheta, Im, segment_threshold)
        plot_image_dict(img, l)
        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments
    

if __name__ == '__main__':
    main()