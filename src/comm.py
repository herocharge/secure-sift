import socket
class Comm:
    def __init__(self, mode='client') -> None:
        self.mode = mode
        self.funcs = {}
        pass

    def start_client(self, host = '127.0.0.1', port = 12346):
        # Create socket object
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to server
        self.client_socket.connect((host, port))
        print(f"Connected to {host}:{port}")

    def register_api(self, name: str, func):
        self.funcs[name] = func
    
    def call_api(self, name, *args):
        self.send_bytes(name)
        for arg in args:
            self.send_bytes(arg)

    def start_server(self, host = '127.0.0.1', port = 12346):
        # Create socket object
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind socket to host and port
        self.server_socket.bind((host, port))

        # Listen for incoming connections
        self.server_socket.listen(5)
        print(f"Server listening on {host}:{port}")

        while True:
            self.client_socket, address = self.server_socket.accept()
            while True:
                api_name = self.receive_bytes().decode()
                if api_name == "exit":
                    break
                if api_name in self.funcs.keys():
                    self.funcs[api_name]()
                else:
                    self.send_bytes(b"Did not find the API you are calling!\n")

            self.client_socket.close()

        
    def send_bytes(self, data, chunk_size=1048576):
        """
        Send bytes over a socket, chunked into `chunk_size`-byte pieces.
        
        :param sock: Socket object
        :param data: Bytes to send
        """
        # Send data length (4-byte integer)
        length = len(data)
        self.client_socket.sendall(length.to_bytes(4, byteorder="big"))
        
        # Send data in chunks
        for i in range(0, length, chunk_size):
            chunk = data[i:i+chunk_size]
            self.client_socket.sendall(chunk)


    def receive_bytes(self, chunk_size=1048576):
        """
        Receive bytes over a socket, chunked into `chunk_size`-byte pieces.
        
        :param sock: Socket object
        :return: Received bytes
        """
        # Receive data length (4-byte integer)
        length_buffer = self.client_socket.recv(4)
        length = int.from_bytes(length_buffer, byteorder="big")
        print(length)
        # Receive data in chunks
        data = b""
        while len(data) < length:
            chunk = self.client_socket.recv(min(chunk_size, length - len(data)))
            data += chunk
        
        return data