import tenseal as ts
from time import time


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

enc_vec = (ts.ckks_vector(context=context, vector=[0.1, 0.2, 0.3]))
enc_vec2 = (ts.ckks_vector(context=context, vector=[0.1, 0.2, 0.3]))

enc_vec.add(enc_vec2)

print(enc_vec.decrypt(secret_key))



