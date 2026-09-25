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





if len(sys.argv) < 1:
    print("No arguments provided. Please pass a image file when running.")
    sys.exit(0)


img = Image.open(sys.argv[1]).convert('L')
img.show()

img_array = np.array(img)

print(createGaussian(5,1))

