import socket
from comm import Comm
import tenseal as ts
import pickle
import numpy as np
from secsift import *
import dill

context = None

def initialize_context(comm):
    global context
    data = comm.receive_bytes()
    print(data[:10], data[-10:])

    context = ts.context_from(data)
    return 0

def double_vec(comm):
    global context
    data = comm.receive_bytes()
    enc = ts.ckks_vector_from(context, data)
    enc = enc + enc 
    comm.send_bytes(enc.serialize())

def load_image(img):
    global context
    enc_img = []
    for row in range(len(img)):
        enc_row = []
        for col in range(len(img[0])):
            enc_row.append(ts.ckks_vector_from(context, img[row][col]))
        enc_img.append(enc_row)
    return np.array(enc_img)

def serialize_pyramid(img):
    global context
    enc_pyramid = []
    for octave in range(len(img)):
        enc_octave = []
        for size in range(len(img[octave])):
            enc_img = []
            for row in range(len(img[octave][size])):
                enc_row = []
                for col in range(len(img[octave][size][row])):
                    enc_row.append((img[octave][size][row][col].serialize()))
                enc_img.append(enc_row)
            enc_octave.append(enc_img)
        enc_pyramid.append(enc_octave)
    return enc_pyramid


def load_pyramid(img):
    global context
    enc_pyramid = []
    for octave in range(len(img)):
        enc_octave = []
        for size in range(len(img[octave])):
            enc_img = []
            for row in range(len(img[octave][size])):
                enc_row = []
                for col in range(len(img[octave][size][row])):
                    enc_row.append(ts.ckks_vector_from(context, img[octave][size][row][col]))
                enc_img.append(enc_row)
            enc_octave.append(np.array(enc_img))
        enc_pyramid.append(enc_octave)
    return enc_pyramid

def passthrough(comm):
    global context
    raw_img = comm.recv_img()
    print("Image received")
    enc_img = load_image(raw_img)
    print("Sending image")
    comm.send_img(enc_img)
    print("Image sent back")

def generateBaseImage(comm):
    global context
    sigma = 1.6
    assumed_blur = 0.5
    enc_img = load_image(comm.recv_img())
    ret_img = secGenerateBaseImage(np.array(enc_img), sigma=sigma, assumed_blur=assumed_blur)
    comm.end_interactive()
    comm.send_img(ret_img[:,:])

def computeNumberOfOctaves(comm : Comm):
    global context
    size = dill.loads(comm.receive_bytes())
    numOctaves = secComputeNumberOfOctaves(size)
    comm.end_interactive()
    comm.send_bytes(dill.dumps(numOctaves))

def generateGaussianKernels(comm : Comm):
    global context
    sigma = dill.loads(comm.receive_bytes())
    num_intervals = dill.loads(comm.receive_bytes())
    gaussian_kernels = secGenerateGaussianKernels(sigma, num_intervals)
    comm.end_interactive()
    comm.send_bytes(dill.dumps(gaussian_kernels))

def generateGaussianImages(comm : Comm):
    global context
    base_image = load_image(dill.loads(comm.receive_bytes()))
    num_octaves = dill.loads(comm.receive_bytes())
    gaussian_kernels = dill.loads(comm.receive_bytes())
    gaussian_kernels = secGenerateGaussianImages(base_image, num_octaves, gaussian_kernels)
    comm.end_interactive()
    comm.send_bytes(dill.dumps(serialize_pyramid(gaussian_kernels)))

def generateDoGImages(comm : Comm):
    global context
    gaussian_images = load_pyramid(dill.loads(comm.receive_bytes()))
    dog_images = secGenerateDoGImages(gaussian_images)
    comm.end_interactive()
    comm.send_bytes(dill.dumps(serialize_pyramid(dog_images)))

def findScaleSpaceExtrema(comm : Comm):
    global context
    gaussian_images = load_pyramid(dill.loads(comm.receive_bytes()))
    dog_images = load_pyramid(dill.loads(comm.receive_bytes()))
    sigma = dill.loads(comm.receive_bytes())
    image_border_width = dill.loads(comm.receive_bytes())
    cmp_count = 0
    refresh_count = 0
    def cmp(x, a, b):
        cmp_count += 1
        x_enc = False
        a_enc = False
        b_enc = False
        if isinstance(x, ts.CKKSVector):
            x = x.serialize()
            x_enc = True
        if isinstance(a, ts.CKKSVector):
            a = a.serialize()
            a_enc = True
        if isinstance(b, ts.CKKSVector):
            b = b.serialize()
            b_enc = True
        comm.send_bytes(b'cmp')
        comm.send_bytes(dill.dumps(x_enc))
        comm.send_bytes(dill.dumps(a_enc))
        comm.send_bytes(dill.dumps(b_enc))
        comm.send_bytes(dill.dumps(x))
        comm.send_bytes(dill.dumps(a))
        comm.send_bytes(dill.dumps(b))
        return ts.ckks_vector_from(context, comm.receive_bytes())
    def refresh(x):
        refresh_count += 1
        comm.send_bytes(b'refresh')
        comm.send_bytes(dill.dumps(x.serialize()))
        return ts.ckks_vector_from(context, comm.receive_bytes())

    secFindScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, cmp=cmp, refresh=np.vectorize(refresh))
    comm.end_interactive()
    print(f"find scale space - cmps: {cmp_count} refs: {refresh_count}")
    comm.send_bytes(dill.dumps(serialize_pyramid(dog_images)))



def start_server():
    comm = Comm(mode='server')
    comm.register_api('init_context', lambda: initialize_context(comm))
    comm.register_api('double_vec', lambda: double_vec(comm))
    comm.register_api('passthrough', lambda: passthrough(comm))
    comm.register_api('generate_base_image', lambda: generateBaseImage(comm))
    comm.register_api('compute_num_octaves', lambda: computeNumberOfOctaves(comm))
    comm.register_api('generate_gaussian_kernels', lambda: generateGaussianKernels(comm))
    comm.register_api('generate_gaussian_images', lambda: generateGaussianImages(comm))
    comm.register_api('generate_dog_images', lambda: generateDoGImages(comm))
    comm.register_api('find_scale_space_extrema', lambda:findScaleSpaceExtrema(comm))
    comm.start_server()

if __name__ == "__main__":
    start_server()
