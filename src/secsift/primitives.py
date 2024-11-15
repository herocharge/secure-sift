import numpy as np
import tenseal as ts

DEBUG = 1

def secENC(val, params = None):
    if DEBUG:
        return val
    
    raise NotImplementedError("Not implemented")

def secAdd(a, b):
    if DEBUG:
        return a + b
    
    raise NotImplementedError("Not implemented")

def secSub(a, b):
    if DEBUG:
        b = secMul(b, -1)
        return secAdd(a, b)
    
    raise NotImplementedError("Not implemented")
    
def secMul(a, b):
    if DEBUG:
        return a * b
    
    raise NotImplementedError("Not implemented")

def secDiv(a, b):
    if DEBUG:
        return a / b

    raise NotImplementedError("Not implemented")

def secCompare(a, b):
    if DEBUG:
        return a > b
    
    raise NotImplementedError("Not implemented")

def secLTSQ(a, b, rcond=None):
    if DEBUG:
        return np.linalg.lstsq(a, b, rcond)
    
    raise NotImplementedError("Not implemented")
