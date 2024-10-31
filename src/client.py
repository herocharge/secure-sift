import socket
import pickle
import tenseal as ts

def start_client():
    host = '127.0.0.1'
    port = 12345

    # Create socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to server
    client_socket.connect((host, port))
    print(f"Connected to {host}:{port}")

    # Create Tenseal context and encrypt data
    context = ts.context(scheme="CKKS", bits_scale=30)
    encoder = ts.encoder(context)
    data = encoder.encode([1.2, 2.4, 3.6])
    encrypted_data = ts.encrypt(data, context)

    # Send encrypted data and context to server
    message = pickle.dumps((encrypted_data, context))
    client_socket.send(message)

    # Receive encrypted data and context from server
    response = client_socket.recv(4096)
    encrypted_data, context = pickle.loads(response)

    # Decrypt received data
    decrypted_data = ts.decrypt(encrypted_data, context)
    print("Decrypted data:", decrypted_data)

    client_socket.close()

if __name__ == "__main__":
    start_client()