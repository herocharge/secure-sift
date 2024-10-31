import socket
from comm import send_bytes, receive_bytes
import tenseal as ts


def start_client():
    host = '127.0.0.1'
    port = 12345

    # Create socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to server
    client_socket.connect((host, port))
    print(f"Connected to {host}:{port}")

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

    encrypted_data = ts.ckks_vector(context, [0.1, 0.2, 0.3])

    # Send encrypted data and context to server
    # message = pickle.dumps((encrypted_data.serialize(), context.serialize()))
    con_ser = context.serialize()

    print(con_ser[:10], con_ser[-10:])
    # context_size = (len(con_ser))
    # print(context_size)
    # client_socket.sendall(struct.pack("!I", context_size) + con_ser)
    send_bytes(client_socket, context.serialize())
    send_bytes(client_socket, encrypted_data.serialize())
    # client_socket.send(encrypted_data.serialize())

    # Receive encrypted data and context from server
    response = receive_bytes(client_socket)
    result = ts.ckks_vector_from(context, response)

    # Decrypt received data
    decrypted_data = result.decrypt(secret_key)
    print("Decrypted data:", decrypted_data)

    client_socket.close()

if __name__ == "__main__":
    start_client()