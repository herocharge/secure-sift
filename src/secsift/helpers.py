from primitives import *

def sec1DVectorProduct(a, b):
    if DEBUG:
        product = np.zeros_like(a, dtype=np.float64)
        for i in range(a.shape[0]):
            product[i] = secMul(a[i], b[i])
        return product
    
    raise NotImplementedError("Not implemented")

def secDot1D(a, b):
    if DEBUG:
        return np.dot(a, b)
    
    raise NotImplementedError("Not implemented")

def sec2DVectorProduct(a, b):
    if DEBUG:
        for i in range(len(a)):
            for j in range(len(a[0])):
                a[i][j] = secMul(a[i][j], b[i][j])

        return a
    
    raise NotImplementedError("Not implemented")

def sec2DVectorSum(a):
    if DEBUG:
        ret = 0
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                ret = secAdd(ret, a[i, j])
        return ret
    
    raise NotImplementedError("Not implemented")


def sec2DVectorProduct(a, b):
    if DEBUG:
        product = np.zeros_like(a, dtype=np.float64)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                product[i, j] = secMul(a[i, j], b[i, j])
        return product
    
    product = np.zeros_like(a, dtype=ts.CKKSVector)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            product[i, j] = secMul(a[i, j], b[i, j])
    return product

def secClip(a, min_val, max_val):
    """
        a is a scalar
    """
    if DEBUG:
        if not secCompare(a, min_val):
            return min_val
        if secCompare(a, max_val):
            return max_val
        return a
    
    raise NotImplementedError("Not implemented")

def secClip2DVetor(a, min_val, max_val):
    if DEBUG:
        for i in range(len(a)):
            for j in range(len(a[0])):
                a[i][j] = secClip(a[i][j], min_val, max_val)

        return a
    
    raise NotImplementedError("Not implemented")

def secConvolve2d(image, kernel, isKernelEncrypted=True):
    """
        Perform a 2D convolution on an encrypted image using an encrypted kernel.
        image and kernel are 2D numpy arrays.
    """
    if DEBUG:
        if not isKernelEncrypted:
            raise NotImplementedError("Kernel must be encrypted")
        
        img_height, img_width = image.shape
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        
        img_padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
        img_out = np.zeros((img_height, img_width), dtype=np.float64)

        for i in range(img_height):
            for j in range(img_width):
                patch = img_padded[i:i + kernel_height, j:j + kernel_width]
                vector_prod = sec2DVectorProduct(patch, kernel)
                img_out[i, j] = sec2DVectorSum(vector_prod)

        img_out = secClip2DVetor(img_out, 0, 255)
        return img_out.astype(np.uint8)

    img_height, img_width, num_channels = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    img_padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
    img_out = np.zeros((img_height, img_width, num_channels), dtype=ts.CKKSVector)

    for c in range(num_channels):
        for i in range(img_height):
            for j in range(img_width):

                patch = img_padded[i:i + kernel_height, j:j + kernel_width, c]
                vector_prod = sec2DVectorProduct(patch, kernel)
                img_out[i, j, c] = sec2DVectorSum(vector_prod)

    img_out = secClip2DVetor(img_out, 0, 255)
    return img_out

def secResize(image, dsize, fx=None, fy=None, interpolation=None):
    if DEBUG:
        if fx is None and fy is None:
            fx = dsize[0] / image.shape[1]
            fy = dsize[1] / image.shape[0]

        print("Image shape: ", image.shape)
        
        new_image = np.zeros((dsize[1], dsize[0]), dtype=np.uint8)
        for y in range(dsize[1]):
            for x in range(dsize[0]):
                new_image[y, x] = image[int(y / fy), int(x / fx)]

        return new_image
    
    if fx is None and fy is None:
        fx = dsize[0] / image.shape[1]
        fy = dsize[1] / image.shape[0]

    print("Image shape: ", image.shape)
    
    new_image = np.zeros((dsize[1], dsize[0], 1), dtype=ts.CKKSVector)
    for y in range(dsize[1]):
        for x in range(dsize[0]):
            new_image[y, x] = image[int(y / fy), int(x / fx)]

    return new_image

def secGaussianBlur(image, kernel_size, sigma):
    if DEBUG:
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()

        new_image = secConvolve2d(image, kernel)
        return new_image

    raise NotImplementedError("Not implemented")

def secSubtract2DVector(a, b):
    if DEBUG:
        out = np.zeros_like(a, dtype=np.float64)
        for i in range(len(a)):
            for j in range(len(a[0])):
                out[i][j] = secSub(a[i][j], b[i][j])

        return out
    
    raise NotImplementedError("Not implemented")

def secCompareAll2DisMinima(image, threshold):
    if DEBUG:
        shape = image.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                # print("Comparing: ", image[i, j], threshold)
                if not secCompare(image[i, j], threshold):
                    return False
        return True
    
    raise NotImplementedError("Not implemented")

def secCompareAll2DisMaxima(image, threshold):
    if DEBUG:
        shape = image.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if secCompare(threshold, image[i, j]):
                    return False
        return True
    
    raise NotImplementedError("Not implemented")

def secTrace(matrix):
    if DEBUG:
        ret = 0
        for i in range(matrix.shape[0]):
            ret = secAdd(ret, matrix[i, i])

        return ret
    
    raise NotImplementedError("Not implemented")

def secDet(matrix):
    if DEBUG:
        if matrix.shape[0] == 2:
            return secSub(secMul(matrix[0, 0], matrix[1, 1]), secMul(matrix[0, 1], matrix[1, 0]))
        else:
            raise NotImplementedError("Not implemented")
    
    raise NotImplementedError("Not implemented")

def secAbs(a):
    if DEBUG:
        return a
    
    raise NotImplementedError("Not implemented")