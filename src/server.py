import socket
from comm import Comm
import tenseal as ts
import pickle
import numpy as np
from secsift import *

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
    return enc_img

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
    comm.send_img(ret_img)

def start_server():
    comm = Comm(mode='server')
    comm.register_api('init_context', lambda: initialize_context(comm))
    comm.register_api('double_vec', lambda: double_vec(comm))
    comm.register_api('passthrough', lambda: passthrough(comm))
    comm.register_api('generate_base_image', lambda: generateBaseImage(comm))
    comm.start_server()

if __name__ == "__main__":
    start_server()