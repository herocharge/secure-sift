import socket
from comm import send_bytes, receive_bytes
import tenseal as ts

def start_server():
    host = '127.0.0.1'
    port = 12345

    # Create socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind socket to host and port
    server_socket.bind((host, port))

    # Listen for incoming connections
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")

    while True:
        client_socket, address = server_socket.accept()
        buf_size = client_socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        print(buf_size)
        print(f"Connected to {address}")


        data = receive_bytes(client_socket)
        print(data[:10], data[-10:])

        context = ts.context_from(data)


        data = receive_bytes(client_socket)
        enc = ts.ckks_vector_from(context, data)

        enc = enc.add(enc)





        # Process encrypted data (e.g., perform computations)
        # For simplicity, just return the same encrypted data

        # Send encrypted data and context back to client
        # response = pickle.dumps()
        send_bytes(client_socket, enc.serialize())
        client_socket.close()

if __name__ == "__main__":
    start_server()