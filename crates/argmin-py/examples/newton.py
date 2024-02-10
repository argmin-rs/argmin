# Copyright 2018-2023 argmin developers
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.
from argmin import Problem, Solver, Executor
import numpy as np
from scipy.optimize import rosen_der, rosen_hess


def main():
    problem = Problem(
        gradient=rosen_der,
        hessian=rosen_hess,
    )
    solver = Solver.Newton
    executor = Executor(problem, solver)
    executor.configure(param=np.array([-1.2, 1.0]), max_iters=8)

    result = executor.run()
    print(result)


if __name__ == "__main__":
    main()
