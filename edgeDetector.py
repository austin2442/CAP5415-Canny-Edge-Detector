from PIL import Image
import numpy as np
import sys

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


# Convolution function, will use zero padding
def convolveX(image, kernel):

    # Iterate through every position in the input
    y_dim, x_dim = image.shape

    kernel_size = kernel.shape[1]

    # edge pad the iamge
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


def computeGradientMagnitude(x_grad, y_grad):
    if x_grad.shape != y_grad.shape:
        raise ValueError("X and Y gradient maps must be same size")

    y_dim, x_dim = x_grad.shape


    magnitude = np.zeros((y_dim,x_dim))
    for y in range(y_dim):
        for x in range(x_dim):
            magnitude[y,x] = np.sqrt(np.square(x_grad[y,x] + np.square(y_grad[y,x])))

    return magnitude

def computeGradientDirection(x_grad, y_grad):
    if x_grad.shape != y_gra.shape:
        raise ValueError("X and Y gradient maps must be same size.")

    y_dim, x_dim = x_grad.shape

    directions = np.zeros((y_dim,x_dim))
    for y in range(y_dim):
        for x in range (x_dim):
            directions[y,x]=np.atan2(y_grad[y,x], x_grad[y,x])
    return directions



def nonMaxSuppression(magnitude,directions ,x_grad, y_grad):
    y_dim, x_dim = gradientMap.shape

    for y in range(y_dim):
        for x in range(x_dim):
            # TODO: Iterate through each pixel, calculate direction then examine two pixels on that direction
            # will probably need to write a liner interpolation function.



if len(sys.argv) < 1:
    print("No arguments provided. Please pass a image file when running.")
    sys.exit(0)


img = Image.open(sys.argv[1]).convert('L')

img_array = np.array(img)

Gaussian = createGaussian(5,1)
DerivativeGaussian = createDerivativeGaussian(5,1)

outputX = convolveX(convolveX(img_array,Gaussian),DerivativeGaussian)
outputY = convolveY(convolveY(img_array,Gaussian.T), DerivativeGaussian.T)

gradientMap = computeGradientMagnitude(outputX, outputY)
directionMap = computeGradientDirection(outputX,outputY)
#Image.fromarray(outputX).show()

#Image.fromarray(outputY).show()
Image.fromarray(gradientMap).show()