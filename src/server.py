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
        for size in range(len(img[0])):
            enc_img = []
            for row in range(len(img[0][0])):
                enc_row = []
                for col in range(len(img[0][0][0])):
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
        for size in range(len(img[0])):
            enc_img = []
            for row in range(len(img[0][0])):
                enc_row = []
                for col in range(len(img[0][0][0])):
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


def start_server():
    comm = Comm(mode='server')
    comm.register_api('init_context', lambda: initialize_context(comm))
    comm.register_api('double_vec', lambda: double_vec(comm))
    comm.register_api('passthrough', lambda: passthrough(comm))
    comm.register_api('generate_base_image', lambda: generateBaseImage(comm))
    comm.register_api('compute_num_octaves', lambda: computeNumberOfOctaves(comm))
    comm.register_api('generate_gaussian_kernels', lambda: generateGaussianKernels(comm))
    comm.register_api('generate_gaussian_images', lambda: generateGaussianImages(comm))
    comm.start_server()

if __name__ == "__main__":
    start_server()
