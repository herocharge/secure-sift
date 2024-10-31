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


secAdd(1, 2)
secSub(1, 2)
secMul(1, 2)
secDiv(float(1), float(2))

def secCompare(a, b):
    if DEBUG:
        return a > b
    
    raise NotImplementedError("Not implemented")

def sec2DVetorProduct(a, b):
    if DEBUG:
        for i in range(len(a)):
            for j in range(len(a[0])):
                a[i][j] = secMul(a[i][j], b[i][j])

        return a
    
    raise NotImplementedError("Not implemented")

def secClip(a, min_val, max_val):
    if DEBUG:
        if secCompare(a, min_val):
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
        image and kernel are 2D numpy arrays
    """
    if isKernelEncrypted == False:
        raise NotImplementedError("Kernel must be encrypted")
    if DEBUG:
        img_height, img_width, num_channels = image.shape
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        
        img_padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
        img_out = np.zeros((img_height, img_width, num_channels))
        
        for c in range(num_channels):
            for i in range(img_height):
                for j in range(img_width):
                    patch = img_padded[i:i+kernel_height, j:j+kernel_width, c]
                    resulting_patch = sec2DVetorProduct(patch, kernel)
                    for x in range(kernel_height):
                        for y in range(kernel_width):
                            img_out[i+x, j+y, c] = resulting_patch[x, y]

        img_out = secClip2DVetor(img_out, 0, 255)
        img_out = img_out.astype(np.uint8) 

        return img_out

    raise NotImplementedError("Not implemented")

def secResize(image, dsize, fx=None, fy=None, interpolation=None):
    if DEBUG:
        if fx is None and fy is None:
            fx = dsize[0] / image.shape[1]
            fy = dsize[1] / image.shape[0]
        
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