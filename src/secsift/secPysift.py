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

def secFindScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04, cmp=None, refresh = lambda x: x):
    """
        Only gaussian_images, dog_images are encrypted
    """

    threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    keypoints = []
    flat_list = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        octave_keypoints = []
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) is the center of the 3x3 array
            img_keypoints = []
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                row_keypoints = []
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    # if isPixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                    keypoint = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, cmp=cmp, refresh = refresh)
                    
                    # keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                    keypoints_with_orientations = [keypoint]*36
                    flat_list.append(keypoints_with_orientations)
                    row_keypoints.append(keypoints_with_orientations)
                img_keypoints.append(row_keypoints)
            octave_keypoints.append(img_keypoints)
        keypoints.append(octave_keypoints)
                    # keypoints.append(keypoint_with_orientation)
    return keypoints, flat_list

class EncKeyPoint:
    def __init__(self, i, j, octave, is_keypoint_present, size, response, angle=None):
        self.i = i
        self.j = j
        self.octave = octave
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

def localizeExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=1, cmp = None, refresh = lambda x: x):
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
    extremum_update = -ltsq_val
    extremum_update = refresh(extremum_update)

    # cmp = lambda x, a, b: a < x < b
    # condition1 =  abs(extremum_update[0]) < 0.5 * denominator and abs(extremum_update[1]) < 0.5 * denominator and abs(extremum_update[2]) < 0.5 * denominator
    condition1 = (cmp(extremum_update[0], -0.5 * denominator, 0.5 * denominator) 
                    * cmp(extremum_update[1], -0.5 * denominator, 0.5 * denominator) 
                    * cmp(extremum_update[2], -0.5 * denominator, 0.5 * denominator))
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] * denominator + 0.5 * np.dot(gradient, extremum_update)
    functionValueAtUpdatedExtremum = refresh(functionValueAtUpdatedExtremum)

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
        i = i,
        j = j,
        octave=octave_index,
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
    tan_right_bins = np.zeros(num_bins // 4) # when angle is -45 to 45 and the (135 to -135, anticlockwise), abs(dx) > abs(dy)
    tan_left_bins = np.zeros(num_bins // 4) # when angle is -45 to 45 and the (135 to -135, anticlockwise), abs(dx) > abs(dy)
    cot_up_bins = np.zeros(num_bins // 4) # otherwise, abs(dx) < abs(dy)
    cot_down_bins = np.zeros(num_bins // 4) # otherwise, abs(dx) < abs(dy)
    
    tan_right_edges = np.linspace(-45, 45, num_bins // 4 + 1)
    tan_left_edges = np.linspace(135, 225, num_bins // 4 + 1)
    cot_up_edges = np.linspace(45, 135, num_bins // 4 + 1)
    cot_down_edges = np.linspace(225, 315, num_bins // 4 + 1)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    # gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    # histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    is_tan = cmp(dy, -dx, dx)
                    is_cot = cmp(dx, -dy, dy)
                    is_tan_right = cmp(dy, 0, 10000)
                    is_tan_left = cmp(dy, -10000, 0)
                    is_cot_up = cmp(dx, 0, 10000)
                    is_cot_down = cmp(dx, -10000, 0)
                    
                    for i, l, r in enumerate(zip(tan_right_edges, tan_right_edges[1:])):
                        cond = cmp(dy, dx * np.tan(np.deg2rad(l)), dx * np.tan(np.deg2rad(r)))
                        tan_right_bins[i] += weight * gradient_magnitude * cond * is_tan_right * is_tan

                    for i, l, r in enumerate(zip(tan_left_edges, tan_left_edges[1:])):
                        cond = cmp(dy, dx * np.tan(np.deg2rad(l)), dx * np.tan(np.deg2rad(r)))
                        tan_left_bins[i] += weight * gradient_magnitude * cond * is_tan_left * is_tan
                    
                    for i, l, r in enumerate(zip(cot_up_edges, cot_up_edges[1:])):
                        cond = cmp(dx, dy * np.cot(np.deg2rad(l)), dy * np.cot(np.deg2rad(r)))
                        cot_up_bins[i] += weight * gradient_magnitude * cond * is_cot_up * is_cot

                    for i, l, r in enumerate(zip(cot_down_edges, cot_down_edges[1:])):
                        cond = cmp(dx, dy * np.cot(np.deg2rad(l)), dy * np.cot(np.deg2rad(r)))
                        cot_down_bins[i] += weight * gradient_magnitude * cond * is_cot_down * is_cot

                    # raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

                    # new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, gradient_orientation, keypoint.response, keypoint.octave)
                    # keypoints_with_orientations.append(new_keypoint)
    
    # This starts at -45 and goes anticlockwise
    bins = np.concatenate((tan_right_bins, cot_up_bins, tan_left_bins, cot_down_bins))
    for n in range(num_bins):
        smooth_histogram[n] = (6 * bins[n] + 4 * (bins[n - 1] + bins[(n + 1) % num_bins]) + bins[n - 2] + bins[(n + 2) % num_bins]) / 16.
    
    def encmax2(a, b):
        cond = cmp(b, a, 10000)
        return (cond) * b + (1 - cond) * (a)
    
    orientation_max = smooth_histogram[0]
    for i in range(1, len(smooth_histogram)):
        orientation_max = encmax2(orientation_max, smooth_histogram[i])
    
    orientation_thresh = peak_ratio * orientation_max
    orientation_peaks = np.zeros(num_bins)
    right_shift_histogram = np.roll(smooth_histogram, 1)
    left_shift_histogram = np.roll(smooth_histogram, -1)
    keypoints_with_orientations = []
    # angles = np.concatenate((ta))
    for i in range(num_bins):
        is_orientation_peak = (cmp(smooth_histogram[i], left_shift_histogram[i], 10000)
                * cmp(smooth_histogram[i], right_shift_histogram[i]))
        is_good_peak_value = cmp(smooth_histogram[i], orientation_thresh, 10000)
        #           interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
        interpolated_peak_index = i
        orientation = 360. - (interpolated_peak_index * 360. / num_bins - 45) # because use started at 45
        if abs(orientation - 360.) < 0.01:
            orientation = 0

        new_keypoint = EncKeyPoint(
            i = keypoint.i,
            j = keypoint.j,
            octave=octave_index,
            is_keypoint_present = is_orientation_peak * is_good_peak_value,
#           interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            size=keypoint.size,
            response=keypoint.response,
            angle=orientation,
        )
        keypoints_with_orientations.append(new_keypoint)
    # orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
    # for peak_index in orientation_peaks:
    #     peak_value = smooth_histogram[peak_index]
    #     if peak_value >= peak_ratio * orientation_max:
    #         # Quadratic peak interpolation
    #         # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    #         left_value = smooth_histogram[(peak_index - 1) % num_bins]
    #         right_value = smooth_histogram[(peak_index + 1) % num_bins]
    #         orientation = 360. - interpolated_peak_index * 360. / num_bins
    #         if abs(orientation - 360.) < float_tolerance:
    #             orientation = 0
    #         new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
    #         keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations

def convertKeypointsToInputImageSize(keypoints):
    """Convert keypoint point, size, and octave to input image size
    """
    for octave_index in range(len(keypoints)):
        for image_index in range(len(keypoints[octave_index])):
            for i in range(len(keypoints[octave_index][image_index])):
                for j in range(len(keypoints[octave_index][image_index][i])):
                    for orientation in range(len(keypoints[octave_index][image_index][i][j])):
                        keypoints[octave_index][image_index][i][j][orientation].i = (0.5 * (keypoints[octave_index][image_index][i][j][orientation].i))
                        keypoints[octave_index][image_index][i][j][orientation].j = (0.5 * (keypoints[octave_index][image_index][i][j][orientation].j))
                        keypoints[octave_index][image_index][i][j][orientation].size *= 0.5
                        keypoints[octave_index][image_index][i][j][orientation].octave = (keypoints[octave_index][image_index][i][j][orientation].octave & ~255) | ((keypoints[octave_index][image_index][i][j][orientation].octave - 1) & 255)
    return keypoints
