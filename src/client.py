import socket
from comm import Comm
import tenseal as ts
import cv2
import dill
import matplotlib.pyplot as plt

def init_enc():
    ## Encryption Parameters

    # controls precision of the fractional part
    bits_scale = 26
    
    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )

    # set the scale
    context.global_scale = pow(2, bits_scale)

    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    secret_key = context.secret_key()
    context.make_context_public()
    return context, secret_key

def encrypt_image(context, img):
    enc_img = []
    for row in range(img.shape[0]):
        enc_row = []
        for col in range(img.shape[1]):
            enc_row.append(ts.ckks_vector(context, img[row][col]))
        enc_img.append(enc_row)
    return enc_img

def encrypt_pyramid(context, img):
    enc_pyramid = []
    for octave in range(len(img)):
        enc_octave = []
        for size in range(len(img[octave])):
            enc_img = []
            for row in range(len(img[octave][size])):
                enc_row = []
                for col in range(len(img[octave][size][row])):
                    enc_row.append(ts.ckks_vector(context, img[octave][size][row][col]))
                enc_img.append(enc_row)
            enc_octave.append(enc_img)
        enc_pyramid.append(enc_octave)
    return enc_pyramid

def load_image(context, img):
    enc_img = []
    for row in range(len(img)):
        enc_row = []
        for col in range(len(img[0])):
            enc_row.append(ts.ckks_vector_from(context, img[row][col]))
        enc_img.append(enc_row)
    return enc_img

def serialize_image(context, img):
    enc_img = []
    for row in range(len(img)):
        enc_row = []
        for col in range(len(img[0])):
            enc_row.append((img[row][col].serialize()))
        enc_img.append(enc_row)
    return enc_img

def load_pyramid(context, img):
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
            enc_octave.append(enc_img)
        enc_pyramid.append(enc_octave)
    return enc_pyramid

def serialize_pyramid(context, img):
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

def show_pyramid(img, secret_key):
    for octave in range(len(img)):
        for size in range(len(img[octave])):
            base_img = ([[int(col.decrypt(secret_key)[0]) for col in row] for row in img[0][0]])
            plt.imshow(base_img)
            plt.show()


def main():
    # Client Side
    comm = Comm()
    comm.start_client()
    context, secret_key = init_enc()

    def cmp(x, a, b, x_enc, a_enc, b_enc):
        x = dill.loads(x)
        a = dill.loads(a)
        b = dill.loads(b)
        if x_enc:
            x = x.decrypt(secret_key)[0]
        if a_enc:
            a = a.decrypt(secret_key)[0]
        if b_enc:
            b = b.decrypt(secret_key)[0]
        return (ts.ckks_vector(context=context, vector=[int(a < x < b)]).serialize())

    def refresh(value):
        value = ts.ckks_vector_from(context, dill.loads(value))
        decrypted_value = value.decrypt(secret_key)[0]
        return (ts.ckks_vector(context=context, vector=[decrypted_value]).serialize())


    img = cv2.imread('../assets/fly-20x20.jpg')
    sigma=1.6
    num_intervals=3
    assumed_blur=0.5
    image_border_width=5

    comm.call_api(b'init_context', context.serialize())

    enc_img = encrypt_image(context, img)

    # Generate Base Image
    comm.call_api(b'generate_base_image')

    print("Sending image...")
    comm.send_img(enc_img)
    print("Image sent...")

    
    comm.check_interactive(cmp=cmp, refresh=refresh)

    base_img_enc = load_image(context, comm.recv_img())

    base_img = ([[int(col.decrypt(secret_key)[0]) for col in row] for row in base_img_enc])
    plt.imshow(base_img)
    plt.show()

    # Get number of octaves
    comm.call_api(b'compute_num_octaves')

    comm.send_bytes(dill.dumps((len(base_img), len(base_img[0]))))

    comm.check_interactive(cmp=cmp, refresh=refresh)

    num_octaves = dill.loads(comm.receive_bytes())  

    print("Number of octaves: ", num_octaves)  

    # Get gaussian kernels
    comm.call_api(b'generate_gaussian_kernels')

    comm.send_bytes(dill.dumps(sigma))
    comm.send_bytes(dill.dumps(num_intervals))

    comm.check_interactive(cmp=cmp, refresh=refresh)

    gaussian_kernel_sizes = dill.loads(comm.receive_bytes())  
    
    print("Gaussian kernel sizes: ", (gaussian_kernel_sizes))  

    # Get Gaussian pyramid
    comm.call_api(b'generate_gaussian_images')

    comm.send_bytes(dill.dumps(serialize_image(context, base_img_enc)))
    comm.send_bytes(dill.dumps(num_octaves))
    comm.send_bytes(dill.dumps(gaussian_kernel_sizes))

    comm.check_interactive(cmp=cmp, refresh=refresh)

    gaussian_pyramid_enc = load_pyramid(context, dill.loads(comm.receive_bytes()))  
    
    # print("Gaussian kernel sizes: ", (gaussian_kernel_sizes)) 
    show_pyramid(gaussian_pyramid_enc, secret_key)
    
    # Get difference of gaussian pyramid
    comm.call_api(b'generate_dog_images')

    comm.send_bytes(dill.dumps(serialize_pyramid(context, gaussian_pyramid_enc)))

    comm.check_interactive(cmp=cmp, refresh=refresh)

    dog_pyramid_enc = load_pyramid(context, dill.loads(comm.receive_bytes()))  
    
    # print("Gaussian kernel sizes: ", (gaussian_kernel_sizes)) 
    show_pyramid(dog_pyramid_enc, secret_key)
    
    # Find scale space extrema
    comm.call_api(b'generate_dog_images')

    comm.send_bytes(dill.dumps(serialize_pyramid(context, gaussian_pyramid_enc)))
    comm.send_bytes(dill.dumps(serialize_pyramid(context, dog_pyramid_enc)))
    comm.send_bytes(dill.dumps(num_intervals))
    comm.send_bytes(dill.dumps(sigma))
    comm.send_bytes(dill.dumps(image_border_width))

    comm.check_interactive(cmp=cmp, refresh=refresh)

    # dog_pyramid_enc = load_pyramid(context, dill.loads(comm.receive_bytes()))  
    
    # print("Gaussian kernel sizes: ", (gaussian_kernel_sizes)) 
    show_pyramid(dog_pyramid_enc, secret_key)



    comm.call_api(b'exit')


    comm.client_socket.close() # TODO: push this to destructor

if __name__ == "__main__":
    main()
