from finitediff import forward_diff
import numpy as np


def f(x):
    return x[0] ** 2 + x[1] ** 2


g = forward_diff(f)

x = np.array([1.0, 2.0])
print(g(x))
