from finitediff import forward_diff, central_diff, forward_jacobian, central_jacobian
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

g = central_diff(f)
x = np.array([1.0, 2.0])
print(g(x))


def op(x):
    return np.array(
        [
            2.0 * (x[1] ** 3 - x[0] ** 2),
            3.0 * (x[1] ** 3 - x[0] ** 2) + 2.0 * (x[2] ** 3 - x[1] ** 2),
            3.0 * (x[2] ** 3 - x[1] ** 2) + 2.0 * (x[3] ** 3 - x[2] ** 2),
            3.0 * (x[3] ** 3 - x[2] ** 2) + 2.0 * (x[4] ** 3 - x[3] ** 2),
            3.0 * (x[4] ** 3 - x[3] ** 2) + 2.0 * (x[5] ** 3 - x[4] ** 2),
            3.0 * (x[5] ** 3 - x[4] ** 2),
        ]
    )


j = forward_jacobian(op)
x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
print(j(x))

j = central_jacobian(op)
x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
print(j(x))


# class NotCallable:
#     pass


# notcallable = NotCallable()

# g = forward_diff(notcallable)
# x = np.array([1.0, 2.0])
# print(g(x))
