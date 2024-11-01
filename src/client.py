import socket
from comm import Comm
import tenseal as ts
import cv2

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

def load_image(context, img):
    enc_img = []
    for row in range(len(img)):
        enc_row = []
        for col in range(len(img[0])):
            enc_row.append(ts.ckks_vector_from(context, img[row][col]))
        enc_img.append(enc_row)
    return enc_img


def main():
    comm = Comm()
    comm.start_client()
    context, secret_key = init_enc()

    img = cv2.imread('../assets/fly-20x20.jpg')
    for row in img:
        for col in img:
            print(col)

    comm.call_api(b'init_context', context.serialize())

    enc_img = encrypt_image(context, img)

    comm.call_api(b'generate_base_image')

    print("Sending image...")
    comm.send_img(enc_img)
    print("Image sent...")

    res_img = load_image(context, comm.recv_img())
    
    for row in res_img:
        for col in row:
            print(col.decrypt(secret_key))

    comm.call_api(b'exit')


    comm.client_socket.close() # TODO: push this to destructor

if __name__ == "__main__":
    main()