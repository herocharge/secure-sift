import socket
from comm import Comm
import tenseal as ts

context = None

def initialize_context(comm):
    global context
    data = comm.receive_bytes()
    print(data[:10], data[-10:])

    context = ts.context_from(data)

def double_vec(comm):
    global context
    data = comm.receive_bytes()
    enc = ts.ckks_vector_from(context, data)
    enc = enc + enc 
    comm.send_bytes(enc.serialize())

def start_server():
    comm = Comm(mode='server')
    comm.register_api('init_context', lambda: initialize_context(comm))
    comm.register_api('double_vec', lambda: double_vec(comm))
    comm.start_server()

if __name__ == "__main__":
    start_server()