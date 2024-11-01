from secsift.primitives import *
from numpy import sqrt, log
# DEBUG=1
def secGenerateBaseImage(image, sigma, assumed_blur, kernel_size=3):
    """
        Only image is encrypted
    """
    if DEBUG:
        image = secResize(image, (100, 100))
        print("Resize done")
        sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
        return secGaussianBlur(image, kernel_size=kernel_size, sigma=sigma_diff)
    
    raise NotImplementedError("Not implemented")

def secComputeNumberOfOctaves(image_shape):
    """
        image_shape is a not encrypted 
    """
    if DEBUG:
        return int(round(log(min(image_shape)) / log(2) - 1))
    
    raise NotImplementedError("Not implemented")

def secGenerateGaussianKernels(sigma, num_intervals):
    """Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper
       WILL RETURN UNENCRYPTED VALUES 
    """
    if DEBUG:
        num_images_per_octave = num_intervals + 3
        k = 2 ** (1. / num_intervals)
        gaussian_kernels = np.zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
        gaussian_kernels[0] = sigma

        for image_index in range(1, num_images_per_octave):
            sigma_previous = (k ** (image_index - 1)) * sigma
            sigma_total = k * sigma_previous
            gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
        return gaussian_kernels
    
    raise NotImplementedError("Not implemented")



def secGenerateGaussianImages(image, num_octaves, kernels):
    """
        Only image, kernels are encrypted
    """
    if DEBUG:
        gaussian_images = []
        for octave_index in range(num_octaves):
            gaussian_images.append([])
            for kernel in kernels:
                gaussian_images[octave_index].append(secGaussianBlur(image, kernel_size=10, sigma=kernel))
            image = secResize(image, (image.shape[1] // 2, image.shape[0] // 2))
            image = np.array(image)
        return gaussian_images
    
    raise NotImplementedError("Not implemented")

def secGenerateDoGImages(gaussian_images):
    """
        List of encrypted images
    """
    if DEBUG:
        dog_images = []

        for gaussian_images_in_octave in gaussian_images:
            dog_images_in_octave = []
            for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
                dog_images_in_octave.append(secSubtract3DVector(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
            dog_images.append(dog_images_in_octave)
        return dog_images
    
    raise NotImplementedError("Not implemented")

 