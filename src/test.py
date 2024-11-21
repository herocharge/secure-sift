import tenseal as ts
import numpy as np
from time import time


## Encryption Parameters

# controls precision of the fractional part
bits_scale = 26

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192 * 2,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)

# set the scale
context.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context.generate_galois_keys()

secret_key = context.secret_key()
context.make_context_public()

enc_vec = (ts.ckks_vector(context=context, vector=[10, 20, 30]))
enc_vec2 = (ts.ckks_vector(context=context, vector=[0.2, 1.2, 0.004]))

# print(np.array([enc_vec, enc_vec2]))


def f(x, n=10):
    result = 0
    sigma = 0.9
    for k in range(n):
        sign = (-1) ** k
        coefficient = 1 / ((2) ** k * np.math.factorial(k) * (sigma ** (2*k)))
        # coefficient = 1
        term = ( (x)) ** (2 * k)
        result += sign * coefficient * term
    return result
    

# print((enc_vec.sub(enc_vec2)).decrypt(secret_key))



