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
            region = region[:,None]

            output[y,x] = np.sum(region * kernel)

    return output




if len(sys.argv) < 1:
    print("No arguments provided. Please pass a image file when running.")
    sys.exit(0)


img = Image.open(sys.argv[1]).convert('L')

img_array = np.array(img)

kernel = createGaussian(5,100)

outputX = convolveX(img_array,kernel )
outputY = convolveY(img_array,kernel.T)

Image.fromarray(outputX).show()

Image.fromarray(outputY).show()