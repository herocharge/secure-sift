import numpy as np

DEBUG = 1

def secAdd(a, b):
    if DEBUG:
        return a + b
    
    raise NotImplementedError("Not implemented")
    

def secSub(a, b):
    if DEBUG:
        return a - b
    
    raise NotImplementedError("Not implemented")


def secMul(a, b):
    if DEBUG:
        return a * b
    
    raise NotImplementedError("Not implemented")

def secDiv(a, b):
    if DEBUG:
        return a / b

    raise NotImplementedError("Not implemented")

def secCompare(a, b):
    if DEBUG:
        return a > b
    
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
    
    raise NotImplementedError("Not implemented")

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

def secClip3DVetor(a, min_val, max_val):
    if DEBUG:
        shape = a.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    a[i, j, k] = secClip(a[i, j, k], min_val, max_val)

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
        
        img_height, img_width, num_channels = image.shape
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        
        img_padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
        img_out = np.zeros((img_height, img_width, num_channels), dtype=np.float64)

        for c in range(num_channels):
            for i in range(img_height):
                for j in range(img_width):

                    patch = img_padded[i:i + kernel_height, j:j + kernel_width, c]
                    vector_prod = sec2DVectorProduct(patch, kernel)
                    img_out[i, j, c] = sec2DVectorSum(vector_prod)

        img_out = secClip3DVetor(img_out, 0, 255)
        return img_out.astype(np.uint8)

    raise NotImplementedError("Not implemented")

def secResize(image, dsize, fx=None, fy=None, interpolation=None):
    if DEBUG:
        if fx is None and fy is None:
            fx = dsize[0] / image.shape[1]
            fy = dsize[1] / image.shape[0]

        print("Image shape: ", image.shape)
        
        new_image = np.zeros((dsize[1], dsize[0], image.shape[2]), dtype=np.uint8)
        for y in range(dsize[1]):
            for x in range(dsize[0]):
                new_image[y, x] = image[int(y / fy), int(x / fx)]

        return new_image
    
    raise NotImplementedError("Not implemented")

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
        for i in range(len(a)):
            for j in range(len(a[0])):
                a[i][j] = secSub(a[i][j], b[i][j])

        return a
    
    raise NotImplementedError("Not implemented")

def secSubtract3DVector(a, b):
    if DEBUG:
        shape = a.shape
        ret = np.array(a)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    ret[i, j, k] = secSub(a[i, j, k], b[i, j, k])
                    
        return ret
    
    raise NotImplementedError("Not implemented")