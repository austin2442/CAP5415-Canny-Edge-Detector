from PIL import Image
import numpy as np
import sys
from scipy import ndimage
from pathlib import Path

#  Saves intermediate images, normalizes and scales them to ensure intensities are in proper range
def save_intermediate_image(img, filename):
    # Normalize array
    arr_min = img.min()
    arr_max = img.max()

    # Avoid division by zero if the image is perfectly flat
    if arr_max - arr_min == 0:
        normalized = np.zeros_like(img_array)
    else:
        normalized = (img - arr_min) / (arr_max-arr_min)

    # scale to 0-255 and cast to 8-bit
    scaled_array = (normalized * 255).astype(np.uint8)

    # Save
    img = Image.fromarray(scaled_array, mode="L")
    img.save(filename)

    


# Creates Gaussian kernel with square size "size" and standard deviation "sigma"
def createGaussian(size, sigma):
    if size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
        
    offset = size // 2
    
    # Create a 1D array of distances from the center
    # For size=5, x will be [-2, -1,  0,  1,  2]
    x = np.arange(-offset, offset + 1)
    
    # Apply the 1D Gaussian formula
    normal = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    kernel = normal * np.exp(-(x**2) / (2.0 * sigma**2))
    
    # Normalize so the values sum to 1
    kernel = kernel / kernel.sum()
    
    # Reshape into a 2D array representing a row vector (1 row, 'size' columns)
    return kernel.reshape(1, size)



# Creates Derivative of Gaussian kernel with square size "size" and standard deviation "sigma"
def createDerivativeGaussian(size,sigma):
    if size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
        
    offset = size // 2
    x = np.arange(-offset, offset + 1)
    
    # Calculate the standard 1D Gaussian
    normal = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    gaussian = normal * np.exp(-(x**2) / (2.0 * sigma**2))
    
    # Apply the first derivative multiplier
    kernel = (-x / sigma**2) * gaussian
    
    # A derivative kernel must sum to exactly 0. 
    # Subtracting the mean corrects any tiny floating-point inaccuracies.
    kernel = kernel - kernel.mean()
    
    # Reshape into a 2D array representing a row vector (1 row, 'size' columns)
    return kernel.reshape(1, size)


# Convolution function, will use zero padding.
def convolveX(image, kernel):

    # Iterate through every position in the input
    y_dim, x_dim = image.shape

    kernel_size = kernel.shape[1]

    # edge pad the image.
    padded_image =np.pad(image, 
                          pad_width=((0,0),(kernel_size//2,kernel_size//2)),
                          mode='edge')

    output = np.zeros((y_dim,x_dim))

    # Loop over original image dimensions
    for y in range(y_dim):
        for x in range(x_dim):
            region = padded_image[y, x : x + kernel_size]

            output[y,x] = np.sum(region * kernel)


    return output

def convolveY(image, kernel):
    y_dim, x_dim = image.shape

    kernel_size = kernel.shape[0]

    padded_image = np.pad(image,
                          pad_width=((kernel_size//2, kernel_size//2), (0,0)),
                          mode='edge')

    output = np.zeros((y_dim,x_dim))

    for y in range(y_dim):
        for x in range(x_dim):
            region = padded_image[y: y + kernel_size, x]
            region = region[:,None] # Make shape (N,1) so the multiplication works

            output[y,x] = np.sum(region * kernel)

    return output


# Takes in x and y gradients, computes gradient magnitudes
def computeGradientMagnitude(x_grad, y_grad):
    if x_grad.shape != y_grad.shape:
        raise ValueError("X and Y gradient maps must be same size")

    y_dim, x_dim = x_grad.shape

    # Compute magnitude
    magnitude = np.zeros((y_dim,x_dim))
    for y in range(y_dim):
        for x in range(x_dim):
            magnitude[y,x] = np.sqrt(np.square(x_grad[y,x]) + np.square(y_grad[y,x]))

    return magnitude

# Takes in x and y gradients, computes gradient directions.
def computeGradientDirection(x_grad, y_grad):
    if x_grad.shape != y_grad.shape:
        raise ValueError("X and Y gradient maps must be same size.")

    y_dim, x_dim = x_grad.shape

    # Compute directions
    directions = np.zeros((y_dim,x_dim))
    for y in range(y_dim):
        for x in range (x_dim):
            radian_angle =np.atan2(y_grad[y,x], x_grad[y,x])

            directions[y,x] =np.degrees(radian_angle) % 360
    return directions

# Performs hysteresis thresholding using "low_thresh" and "high_thresh" as the thresholds
def hysteresis_thresholding(image, low_thresh, high_thresh):

    # Create Boolean masks

    # Boolean mask of all pixels above high treshold
    strong_edges = image > high_thresh

    # Boolean mask of all pixels above low threshold
    all_edges = image > low_thresh

    # Define 8-way connectivity, used for scipy.ndimage.label
    structure = np.ones((3,3), dtype=int)

    # Calculated connected components. Each component is given a unique id and the pixels of labeled_edges are marked with that id
    labeled_edges, num_features = ndimage.label(all_edges,structure=structure)


    # We find all the labels (connected edges) that are connected to some strong edge
    valid_labels = np.unique(labeled_edges[strong_edges])

    # If the label is one of the strong edges, keep that pixel
    final_edge = np.isin(labeled_edges,valid_labels)


    return (final_edge * 255).astype(np.uint8)





# Performs Non-max Suppression on the gradient map.
def nonMaxSuppression(magnitude, directions):
    y_dim, x_dim = magnitude.shape
    output = np.zeros((y_dim, x_dim))

    # We iterate to y_dim -1 and x_dim -1 to deal the border pixels
    for y in range(1, y_dim-1):
        for x in range(1, x_dim-1):

            # We can reduce number of cases to check by clamping direction to 0 - 180 degrees
            clampedDirection = directions[y,x] % 180 

            # For each of the 4 cases, determine the pixels we will interpolate with and how to weigh each pixel
            if clampedDirection >= 0 and clampedDirection < 45:
                weight = np.tan(np.deg2rad(clampedDirection))
                forwardPixel1, forwardPixel2 = (y, x+1), (y+1, x+1)
                backwardPixel1, backwardPixel2 = (y, x-1), (y-1, x-1)

            elif clampedDirection >= 45 and clampedDirection < 90:
                weight = 1.0 / np.tan(np.deg2rad(clampedDirection))
                forwardPixel1, forwardPixel2 = (y+1, x), (y+1, x+1)
                backwardPixel1, backwardPixel2 = (y-1, x), (y-1, x-1)

            elif clampedDirection >= 90 and clampedDirection < 135:
                weight = abs(1.0 / np.tan(np.deg2rad(clampedDirection)))
                forwardPixel1, forwardPixel2 = (y+1, x), (y+1, x-1)
                backwardPixel1, backwardPixel2 = (y-1, x), (y-1, x+1)

            elif clampedDirection >= 135 and clampedDirection <= 180:
                weight = abs(np.tan(np.deg2rad(clampedDirection)))
                forwardPixel1, forwardPixel2 = (y, x-1), (y+1, x-1)
                backwardPixel1, backwardPixel2 = (y, x+1), (y-1, x+1)

            # Do interpolation
            forwardMagnitude = (1 - weight) * magnitude[forwardPixel1] + weight * magnitude[forwardPixel2]
            backwardMagnitude = (1 - weight) * magnitude[backwardPixel1] + weight * magnitude[backwardPixel2]

            # If our pixel is larger than both forward and back, keep it, otherwise make it zero. 
            if magnitude[y,x] >= forwardMagnitude and magnitude[y,x] >= backwardMagnitude:
                output[y,x] = magnitude[y,x]
            else:
                output[y,x] = 0

    return output


# Function that does full Canny Edge Detection
# image_arr: Input image as a NumPy Array
# kernel_size: Size for square Gaussian kernels
# sigma: Standard deviation for Gaussian kernels
# low_thresh: Low threshold for Hysteresis Thresholding
# high_thresh: High threshold for Hysteresis Thresholding
def CannyEdge(image_arr, kernel_size, sigma, low_thresh, high_thresh):

    # Create kernels
    Gaussian = createGaussian(kernel_size,sigma)
    DerivativeGaussian = createDerivativeGaussian(kernel_size,sigma)

    # Do Smoothing
    xSmooth = convolveX(image_arr, Gaussian)
    ySmooth = convolveY(image_arr, Gaussian.T)

    save_intermediate_image(xSmooth, "x_smooth.png")
    save_intermediate_image(ySmooth, "y_smooth.png")

    # Smooth again with derivative kernel
    xGrad = convolveX(xSmooth, DerivativeGaussian)
    yGrad = convolveY(ySmooth, DerivativeGaussian.T)

    save_intermediate_image(xGrad, "x_grad.png")
    save_intermediate_image(yGrad, "y_grad.png")

    # Calculate gradient magnitudes
    gradientMap = computeGradientMagnitude(xGrad, yGrad)

    save_intermediate_image(gradientMap, "grad_map.png")

    # Calculate directions of gradients
    directionMap = computeGradientDirection(xGrad,yGrad)

    # Do non-max surpression
    nonMaxSuppressionOutput = nonMaxSuppression(gradientMap,directionMap)

    # Do Hysteresis Thresholding
    final_res = hysteresis_thresholding(nonMaxSuppressionOutput,low_thresh, high_thresh)
    return final_res




if len(sys.argv) < 1:
    print("No arguments provided. Please pass a image file when running.")
    sys.exit(0)

img = Image.open(sys.argv[1]).convert('L')
img_array = np.array(img)

res = CannyEdge(img_array,5,1,5,20)
res_img = Image.fromarray(res)
res_img.save("output.png")