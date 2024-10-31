import socket
import pickle

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
        print(f"Connected to {address}")

        # Receive encrypted data and context from client
        data = client_socket.recv(4096)
        encrypted_data, context = pickle.loads(data)

        # Process encrypted data (e.g., perform computations)
        # For simplicity, just return the same encrypted data

        # Send encrypted data and context back to client
        response = pickle.dumps((encrypted_data, context))
        client_socket.send(response)

        client_socket.close()

if __name__ == "__main__":
    start_server()