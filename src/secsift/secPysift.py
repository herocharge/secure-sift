from primitives import *
from primitives import *
from numpy import sqrt, log
from tqdm import tqdm

# DEBUG=1
def secGenerateBaseImage(image, sigma, assumed_blur, kernel_size=3):
    """
        Only image is encrypted
    """
    if DEBUG:
        # image = secResize(image, (20, 20))
        print("Resize done")
        sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
        return secGaussianBlur(image, kernel_size=kernel_size, sigma=sigma_diff)
    
    # image = secResize(image, (20, 20))
    print("Resize done")
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return secGaussianBlur(image, kernel_size=kernel_size, sigma=sigma_diff)

def secComputeNumberOfOctaves(image_shape):
    """
        image_shape is a not encrypted 
    """
    if DEBUG:
        return int(round(log(min(image_shape)) / log(2) - 1))
    
    return int(round(log(min(image_shape)) / log(2) - 1))

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
    
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = np.zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels


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
    
    gaussian_images = []
    for octave_index in tqdm(range(num_octaves)):
        gaussian_images.append([])
        for kernel in kernels:
            gaussian_images[octave_index].append(secGaussianBlur(image, kernel_size=10, sigma=kernel))
        image = secResize(image, (image.shape[1] // 2, image.shape[0] // 2))
    return gaussian_images

def secGenerateDoGImages(gaussian_images):
    """
        List of encrypted images
    """
    if DEBUG:
        dog_images = []

        for gaussian_images_in_octave in gaussian_images:
            dog_images_in_octave = []
            for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
                dog_images_in_octave.append(secSubtract2DVector(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
                dog_images_in_octave.append(secSubtract2DVector(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
            dog_images.append(dog_images_in_octave)
        return dog_images
    
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(secSubtract2DVector(second_image, first_image))
        dog_images.append(dog_images_in_octave)
    return dog_images

def secFindScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04, cmp=None):
    """Find pixel positions of all scale-space extrema in the image pyramid
    """

    threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        octave_keypoints = []
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) is the center of the 3x3 array
            img_keypoints = []
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                row_keypoints = []
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    # if isPixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                    keypoint = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, cmp=cmp)
                    
                    # keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                    keypoints_with_orientations = [keypoint]*36
                    row_keypoints.append(keypoints_with_orientations)
                img_keypoints.append(row_keypoints)
            octave_keypoints.append(img_keypoints)
        keypoints.append(octave_keypoints)
                    # keypoints.append(keypoint_with_orientation)
    return keypoints

class EncKeyPoint:
    def __init__(self, is_keypoint_present, size, response, angle=None):
        self.is_keypoint_present = is_keypoint_present
        self.size = size
        self.response = response
        self.angle = angle
    


def isPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
    """
        threshold is not encrypted
        Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise
    """
    center_pixel_value = second_subimage[1, 1]
    
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return (center_pixel_value >= first_subimage).all() and \
                   (center_pixel_value >= third_subimage).all() and \
                   (center_pixel_value >= second_subimage[0, :]).all() and \
                   (center_pixel_value >= second_subimage[2, :]).all() and \
                   center_pixel_value >= second_subimage[1, 0] and \
                   center_pixel_value >= second_subimage[1, 2]
        elif center_pixel_value < 0:
            return (center_pixel_value <= first_subimage).all() and \
                   (center_pixel_value <= third_subimage).all() and \
                   (center_pixel_value <= second_subimage[0, :]).all() and \
                   (center_pixel_value <= second_subimage[2, :]).all() and \
                   center_pixel_value <= second_subimage[1, 0] and \
                   center_pixel_value <= second_subimage[1, 2]

    # if secComparePixel(abs(center_pixel_value), threshold):
    #     if secComparePixel(center_pixel_value, 0):
    #         return secCompareAll2DisMinima(first_subimage, center_pixel_value) and \
    #                secCompareAll2DisMinima(third_subimage, center_pixel_value) and \
    #                secCompareAll2DisMinima(second_subimage[0, :], center_pixel_value) and \
    #                secCompareAll2DisMinima(second_subimage[2, :], center_pixel_value) and \
    #                secComparePixel(center_pixel_value, second_subimage[1, 0]) and \
    #                secComparePixel(center_pixel_value , second_subimage[1, 2])
    #     elif secComparePixel(0, center_pixel_value):
    #         return secCompareAll2DisMaxima(first_subimage, center_pixel_value) and \
    #                secCompareAll2DisMaxima(third_subimage, center_pixel_value) and \
    #                secCompareAll2DisMaxima(second_subimage[0, :], center_pixel_value) and \
    #                secCompareAll2DisMaxima(second_subimage[2, :], center_pixel_value) and \
    #                secComparePixel(center_pixel_value, second_subimage[1, 0]) and \
    #                secComparePixel(center_pixel_value , second_subimage[1, 2])
    return False

def localizeExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=1, cmp = None):
    """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
    """
    print("Num attempts: ", num_attempts_until_convergence)
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    extremum_update = np.array([1, 1, 1], dtype='float32')  # (di, dj, ds)
        # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
    first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
    pixel_cube = np.stack([first_image[i-1:i+2, j-1:j+2],
                        second_image[i-1:i+2, j-1:j+2],
                        third_image[i-1:i+2, j-1:j+2]]) * (1 / 255)
    gradient = computeGradientAtCenterPixel(pixel_cube)
    hessian = computeHessianAtCenterPixel(pixel_cube)
    ltsq_val, denominator = secLTSQ(hessian, gradient)
    # ltsq_val = ltsq_val[0]
    extremum_update = secMul(-1, ltsq_val)
    # cmp = lambda x, a, b: a < x < b
    # condition1 =  abs(extremum_update[0]) < 0.5 * denominator and abs(extremum_update[1]) < 0.5 * denominator and abs(extremum_update[2]) < 0.5 * denominator
    condition1 = (cmp(extremum_update[0], -0.5 * denominator, 0.5 * denominator) 
                    * cmp(extremum_update[1], -0.5 * denominator, 0.5 * denominator) 
                    * cmp(extremum_update[2], -0.5 * denominator, 0.5 * denominator))
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] * denominator + 0.5 * np.dot(gradient, extremum_update)
    
    # condition2 = abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold * denominator
    condition2 = -cmp(functionValueAtUpdatedExtremum * num_intervals, -contrast_threshold * denominator, -contrast_threshold * denominator) + 1
    
    xy_hessian = hessian[:2, :2]
    xy_hessian_trace = secTrace(xy_hessian)
    xy_hessian_det = secDet2x2(xy_hessian)
    # condition3 =  xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det
    condition3 = cmp(xy_hessian_det, 0, 10000) * cmp(eigenvalue_ratio * (xy_hessian_trace ** 2), -10000, ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det)
    # Contrast check passed -- construct and return OpenCV KeyPoint object
    # Ignoring extram_update[2] in size, because it will be of the order 1 - 1.2 which is negligible
    keypoint = EncKeyPoint(
        is_keypoint_present = condition1 * condition2 * condition3,
        size = sigma * (2 ** ((image_index) / (num_intervals))) * (2 ** (octave_index + 1)),
        # response= secAbs(functionValueAtUpdatedExtremum) # check if needed later
        response= (functionValueAtUpdatedExtremum)
    )
    # keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
    # keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
    # keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
    # keypoint.response = secAbs(functionValueAtUpdatedExtremum)
    return keypoint
    # return None

def computeGradientAtCenterPixel(pixel_array):
    """Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
    # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return np.array([dx, dy, ds])

def computeHessianAtCenterPixel(pixel_array):
    """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return np.array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

# #########################
# # Keypoint orientations #
# #########################

def computeKeypointsWithOrientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """Compute orientations for each keypoint
    """
    # logger.debug('Computing keypoint orientations...')
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

                    new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, gradient_orientation, keypoint.response, keypoint.octave)
                    keypoints_with_orientations.append(new_keypoint)

    # for n in range(num_bins):
    #     smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    # orientation_max = max(smooth_histogram)
    # orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
    # for peak_index in orientation_peaks:
    #     peak_value = smooth_histogram[peak_index]
    #     if peak_value >= peak_ratio * orientation_max:
    #         # Quadratic peak interpolation
    #         # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    #         left_value = smooth_histogram[(peak_index - 1) % num_bins]
    #         right_value = smooth_histogram[(peak_index + 1) % num_bins]
    #         interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
    #         orientation = 360. - interpolated_peak_index * 360. / num_bins
    #         if abs(orientation - 360.) < float_tolerance:
    #             orientation = 0
    #         new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
    #         keypoints_with_orientations.append(new_keypoint)
    # return keypoints_with_orientations
