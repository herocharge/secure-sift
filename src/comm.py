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
        # print(length)
        # Receive data in chunks
        data = b""
        while len(data) < length:
            chunk = self.client_socket.recv(min(chunk_size, length - len(data)))
            data += chunk
        
        return data

    def send_img(self, img):
        self.send_bytes(bytes(str(len(img)), encoding='ascii'))
        self.send_bytes(bytes(str(len(img[0])), encoding='ascii'))
        for row in img:
            for col in row:
                self.send_bytes(col.serialize())
                
    def recv_img(self):
        n_rows = int(self.receive_bytes().decode())
        n_cols = int(self.receive_bytes().decode())
        img = []
        for row in range(n_rows):
            recv_row = []
            for _ in range(n_cols):
                recv_row.append(self.receive_bytes())
            img.append(recv_row)
        return img
    
    
    def check_interactive(self, cmp=None, refresh=None):
        while True:
            api_name = self.receive_bytes().decode()
            # print(api_name)
            if api_name == 'end_interaction':
                break
            elif api_name == 'cmp':
                x = self.receive_bytes().decode()
                a = self.receive_bytes().decode()
                b = self.receive_bytes().decode()
                self.send_bytes(cmp(x, a, b))
            elif api_name == 'refresh':
                x = self.receive_bytes().decode()
                self.send_bytes(refresh(x))
        

    def end_interactive(self):
        self.send_bytes(b'end_interaction')
