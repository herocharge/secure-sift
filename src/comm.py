import socket

def send_bytes(sock, data, chunk_size=1048576):
    """
    Send bytes over a socket, chunked into `chunk_size`-byte pieces.
    
    :param sock: Socket object
    :param data: Bytes to send
    """
    # Send data length (4-byte integer)
    length = len(data)
    sock.sendall(length.to_bytes(4, byteorder="big"))
    
    # Send data in chunks
    for i in range(0, length, chunk_size):
        chunk = data[i:i+chunk_size]
        sock.sendall(chunk)


def receive_bytes(sock, chunk_size=1048576):
    """
    Receive bytes over a socket, chunked into `chunk_size`-byte pieces.
    
    :param sock: Socket object
    :return: Received bytes
    """
    # Receive data length (4-byte integer)
    length_buffer = sock.recv(4)
    length = int.from_bytes(length_buffer, byteorder="big")
    print(length)
    # Receive data in chunks
    data = b""
    while len(data) < length:
        chunk = sock.recv(min(chunk_size, length - len(data)))
        data += chunk
    
    return data