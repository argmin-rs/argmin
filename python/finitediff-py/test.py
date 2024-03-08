from finitediff import (
    forward_diff,
    central_diff,
    forward_jacobian,
    central_jacobian,
    forward_jacobian_vec_prod,
    central_jacobian_vec_prod,
    forward_hessian,
    central_hessian,
    forward_hessian_vec_prod,
    central_hessian_vec_prod,
    forward_hessian_nograd,
)
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

j = forward_jacobian_vec_prod(op)
x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
p = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(j(x, p))

j = central_jacobian_vec_prod(op)
x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
p = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(j(x, p))


def f(x):
    return x[0] + x[1] ** 2 + x[2] * x[3] ** 2


def g(x):
    return np.array([1.0, 2.0 * x[1], x[3] ** 2, 2.0 * x[3] * x[2]])


h = forward_hessian(g)
x = np.array([1.0, 1.0, 1.0, 1.0])
print(h(x))

h = central_hessian(g)
x = np.array([1.0, 1.0, 1.0, 1.0])
print(h(x))

h = forward_hessian_vec_prod(g)
x = np.array([1.0, 1.0, 1.0, 1.0])
p = np.array([2.0, 3.0, 4.0, 5.0])
print(h(x, p))

h = central_hessian_vec_prod(g)
x = np.array([1.0, 1.0, 1.0, 1.0])
p = np.array([2.0, 3.0, 4.0, 5.0])
print(h(x, p))

h = forward_hessian_nograd(f)
x = np.array([1.0, 1.0, 1.0, 1.0])
print(h(x))
# class NotCallable:
#     pass


# notcallable = NotCallable()

# g = forward_diff(notcallable)
# x = np.array([1.0, 2.0])
# print(g(x))
