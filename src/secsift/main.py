import cv2

from secPysift import *

img1 = cv2.imread('box.png')  

base_image = secGenerateBaseImage(img1, 1.6, 0.5)
num_octaves = secComputeNumberOfOctaves(base_image.shape)
gaussian_kernels = secGenerateGaussianKernels(1.6, 3)
gaussian_images = secGenerateGaussianImages(base_image, num_octaves, gaussian_kernels)
dog_images = secGenerateDoGImages(gaussian_images)

