from finitediff import forward_diff
import numpy as np


def f(x):
    return x[0] ** 2 + x[1] ** 2


g = forward_diff(f)

x = np.array([1.0, 2.0])
print(g(x))


class Blubb:
    def __call__(self, x):
        return x[0] ** 2 + x[1] ** 2

    def blaah(self, x):
        return x[0] ** 2 + x[1] ** 2


pf = Blubb()

g = forward_diff(pf)
x = np.array([1.0, 2.0])
print(g(x))

g = forward_diff(pf.blaah)
x = np.array([1.0, 2.0])
print(g(x))


class NotCallable:
    pass


notcallable = NotCallable()

g = forward_diff(notcallable)
x = np.array([1.0, 2.0])
print(g(x))
