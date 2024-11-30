import tenseal as ts
import dill

class CustomNumber:
    def __init__(self, cmp=None, func=lambda secret_key: 0):
        self.cmp = cmp
        if cmp is not None:
            self.cmp[1] = self.cmp[1].serialize()
            self.cmp[2] = self.cmp[2].serialize()
            # print(type(self.cmp[1]))
        self.func = func


    def eval_cmp(self, secret_key):
        if self.cmp[0] == '<':
            return ts.ckks_vector_from(secret_key[1], self.cmp[1]).decrypt(secret_key[0])[0] < ts.ckks_vector_from(secret_key[1], self.cmp[2]).decrypt(secret_key[0])[0]
        return NotImplemented

    def __call__(self, secret_key):
        if self.cmp is not None:
            b = self.eval_cmp(secret_key)   
            return self.func(b, secret_key)
        val = self.func(secret_key)
        if isinstance(val, ts.CKKSVector):
            return val.decrypt(secret_key[0])[0]
        return val

    def __repr__(self):
        return f"CustomNumber({self.cmp})"

    # Addition
    def __add__(self, other):
        if isinstance(other, CustomNumber):
            return CustomNumber(func=lambda secret_key: self(secret_key) + other(secret_key))
        tmp = CustomNumber(func=lambda secret_key:other)
        return CustomNumber(func=lambda secret_key: self(secret_key) + tmp(secret_key))

    def __radd__(self, other):
        return self + other

    # Subtraction
    def __sub__(self, other):
        if isinstance(other, CustomNumber):
            return CustomNumber(func=lambda secret_key: self(secret_key) - other(secret_key))
        tmp = CustomNumber(func=lambda secret_key:other)
        return CustomNumber(func=lambda secret_key: self(secret_key) - tmp(secret_key))

    def __rsub__(self, other):
        if isinstance(other, CustomNumber):
            return CustomNumber(func=lambda secret_key: other(secret_key) - self(secret_key))
        tmp = CustomNumber(func=lambda secret_key:other)
        return CustomNumber(func=lambda secret_key: tmp(secret_key) - self(secret_key))

    # Multiplication
    def __mul__(self, other):
        if isinstance(other, CustomNumber):
            return CustomNumber(func=lambda secret_key: self(secret_key) * other(secret_key))
        tmp = CustomNumber(func=lambda secret_key:other)
        return CustomNumber(func=lambda secret_key: self(secret_key) * tmp(secret_key))

    def __rmul__(self, other):
        return self * other
    
    # Comparison
    # def __eq__(self, other):
    #     if isinstance(other, CustomNumber):
    #         return self.value == other.value
    #     return NotImplemented

    # def __ne__(self, other):
    #     return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, CustomNumber):
            return CustomNumber()
        return NotImplemented

    # def __le__(self, other):
    #     return self < other or self == other

    # def __gt__(self, other):
    #     return not self <= other

    # def __ge__(self, other):
    #     return not self < other


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

# Example usage:
v1 = ts.ckks_vector(context=context, vector=[0.1])
v2 = ts.ckks_vector(context=context, vector=[0.5])
v3 = ts.ckks_vector(context=context, vector=[1])
v4 = ts.ckks_vector(context=context, vector=[2])
# v1 = 10
# v2 = 11
# v3 = 1
# v4 = 2
a = CustomNumber(func=lambda x: v1)
b = CustomNumber(func=lambda x: v2)

c = CustomNumber(cmp=['<', v2, v1], func=lambda b, x: b * v3 + (1 - b)*v4)
# c = CustomNumber(cmp=('<', v2, v1), func=lambda b, x: b)

# def custom_deserialize(obj):
#     obj.__setstate__(obj.__getstate__(), context)
#     return obj

# dill.pickles.register(CustomNumber, custom_deserialize)
# dill.pickles.register(CustomNumber, custom_deserialize)
# def dumper(self):
#     print(self)
#     # Return a dictionary representing the object's state
#     if self.cmp is None:
#         return {"cmp" : None, "func" : self.func}
#     c1 = self.cmp[1]
#     if isinstance(self.cmp[1], ts.CKKSVector):
#         c1.serialize()
#     c2 = self.cmp[2]
#     if isinstance(self.cmp[2], ts.CKKSVector):
#         c2.serialize()
    
#     return {"cmp": ((self.cmp[0], c1, c2)), "func": (self.func)}
# dill.pickle(CustomNumber, dumper)

# @dill.register(CustomNumber)
# def loader(state, self):
#     print(state.items, self, context)
#     # Restore the object's state from the dictionary
#     self.cmp = state["cmp"]
#     if self.cmp is not None:
#         self.cmp[1] = ts.ckks_vector_from(context, self.cmp[1])
#         self.cmp[2] = ts.ckks_vector_from(context, self.cmp[2])

#     self.func = state["func"]


d = c + a  # Trivial case: a + 0
print(d((secret_key, context)))  # CustomNumber(5)



serialized_num = dill.dumps(d)
obj = dill.loads(serialized_num)
print((serialized_num))
print(obj((secret_key, context)))
# serialized_num = dill.dumps(d)

# e = a * c
# print(e)  # CustomNumber(10)

# f = a - b  # Trivial case: a - 0
# print(f)  # CustomNumber(5)

# g = a > c
# print(g)  # True