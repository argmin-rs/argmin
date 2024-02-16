<p align="center">
  <img
    width="400"
    src="https://raw.githubusercontent.com/argmin-rs/argmin/main/media/logo.png"
  />
</p>
<h1 align="center">argmin-testfunctions-py</h1>

<p align="center">
  <a href="https://argmin-rs.org">Website</a>
  |
  <a href="https://argmin-rs.org/book/">Book</a>
  |
  <a href="https://docs.rs/argmin_testfunctions">Docs (latest release)</a>
  |
  <a href="https://argmin-rs.github.io/argmin/argmin_testfunctions/index.html">Docs (main branch)</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/argmin-testfunctions-py/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/argmin-testfunctions-py?style=flat-square">
  </a>
  <a href="https://github.com/argmin-rs/argmin/actions"
    ><img
      src="https://img.shields.io/github/actions/workflow/status/argmin-rs/argmin/python.yml?branch=main&label=argmin CI&style=flat-square"
      alt="GitHub Actions workflow status"
  /></a>
  <img
    src="https://img.shields.io/crates/l/argmin?style=flat-square"
    alt="License"
  />
  <a href="https://discord.gg/fYB8AwxxMW"
    ><img
      src="https://img.shields.io/discord/1189119565335109683?style=flat-square&label=argmin%20Discord"
      alt="argmin Discord"
  /></a>
</p>

This Python module makes the test functions of the `argmin_testfunctions` Rust crate available in Python. 
For each test function the derivative and Hessian are available as well. 
While most functions are two-dimensional, some allow an arbitrary number of parameters.
For some functions additional optional parameters are accessible, which can be used to modify the shape of the test function.
For details on the individual test functions please consult the docs of the Rust library, either for the
[latest release](https://docs.rs/argmin_testfunctions) or the
[current main branch](https://argmin-rs.github.io/argmin/argmin_testfunctions/index.html).

## Examples

```python
from argmin_testfunctions_py import *

# Ackley (arbitrary number of parameters)
c = ackley([0.1, 0.2, 0.3, 0.4])
g = ackley_derivative([0.1, 0.2, 0.3, 0.4])
h = ackley_hessian([0.1, 0.2, 0.3, 0.4])

# Ackley with custom (optional) parameters a, b, and c.
c = ackley([0.1, 0.2, 0.3, 0.4], a = 10.0, b = 0.3, c = 3.14)
g = ackley_derivative([0.1, 0.2, 0.3, 0.4], a = 10.0, b = 0.3, c = 3.14)
h = ackley_hessian([0.1, 0.2, 0.3, 0.4], a = 10.0, b = 0.3, c = 3.14)

# Beale
c = beale([0.1, 0.2])
g = beale_derivative([0.1, 0.2])
h = beale_hessian([0.1, 0.2])

# Booth
c = booth([0.1, 0.2])
g = booth_derivative([0.1, 0.2])
h = booth_hessian([0.1, 0.2])

# Bukin No. 6
c = bukin_n6([0.1, 0.2])
g = bukin_n6_derivative([0.1, 0.2])
h = bukin_n6_hessian([0.1, 0.2])

# Cross-in-tray
c = cross_in_tray([0.1, 0.2])
g = cross_in_tray_derivative([0.1, 0.2])
h = cross_in_tray_hessian([0.1, 0.2])

# Easom
c = easom([0.1, 0.2])
g = easom_derivative([0.1, 0.2])
h = easom_hessian([0.1, 0.2])

# Eggholder
c = eggholder([0.1, 0.2])
g = eggholder_derivative([0.1, 0.2])
h = eggholder_hessian([0.1, 0.2])

# Goldstein-Price
c = goldsteinprice([0.1, 0.2])
g = goldsteinprice_derivative([0.1, 0.2])
h = goldsteinprice_hessian([0.1, 0.2])

# Himmelblau
c = himmelblau([0.1, 0.2])
g = himmelblau_derivative([0.1, 0.2])
h = himmelblau_hessian([0.1, 0.2])

# Holder-Table
c = holder_table([0.1, 0.2])
g = holder_table_derivative([0.1, 0.2])
h = holder_table_hessian([0.1, 0.2])

# Levy (arbitrary number of parameters)
c = levy([0.1, 0.2, 0.3, 0.4])
g = levy_derivative([0.1, 0.2, 0.3, 0.4])
h = levy_hessian([0.1, 0.2, 0.3, 0.4])

# Levy No. 13
c = levy_n13([0.1, 0.2])
g = levy_n13_derivative([0.1, 0.2])
h = levy_n13_hessian([0.1, 0.2])

# Matyas
c = matyas([0.1, 0.2])
g = matyas_derivative([0.1, 0.2])
h = matyas_hessian([0.1, 0.2])

# McCorminck
c = mccorminck([0.1, 0.2])
g = mccorminck_derivative([0.1, 0.2])
h = mccorminck_hessian([0.1, 0.2])

# Picheny
c = picheny([0.1, 0.2])
g = picheny_derivative([0.1, 0.2])
h = picheny_hessian([0.1, 0.2])

# Rastrigin (with arbitrary number of parameters)
c = rastrigin([0.1, 0.2, 0.3, 0.4])
g = rastrigin_derivative([0.1, 0.2, 0.3, 0.4])
h = rastrigin_hessian([0.1, 0.2, 0.3, 0.4])

# Rastrigin with custom (optional) parameter a.
c = rastrigin([0.1, 0.2, 0.3, 0.4], a = 5.0)
g = rastrigin_derivative([0.1, 0.2, 0.3, 0.4], a = 5.0)
h = rastrigin_hessian([0.1, 0.2, 0.3, 0.4], a = 5.0)

# Rosenbrock (with arbitrary number of parameters)
c = rosenbrock([0.1, 0.2, 0.3, 0.4])
g = rosenbrock_derivative([0.1, 0.2, 0.3, 0.4])
h = rosenbrock_hessian([0.1, 0.2, 0.3, 0.4])

# Rosenbrock with custom (optional) parameters a and b.
c = rosenbrock([0.1, 0.2, 0.3, 0.4], a = 5.0, b = 200.0)
g = rosenbrock_derivative([0.1, 0.2, 0.3, 0.4], a = 5.0, b = 200.0)
h = rosenbrock_hessian([0.1, 0.2, 0.3, 0.4], a = 5.0, b = 200.0)

# Schaffer No. 2
c = schaffer_n2([0.1, 0.2])
g = schaffer_n2_derivative([0.1, 0.2])
h = schaffer_n2_hessian([0.1, 0.2])

# Schaffer No. 4
c = schaffer_n4([0.1, 0.2])
g = schaffer_n4_derivative([0.1, 0.2])
h = schaffer_n4_hessian([0.1, 0.2])

# Sphere (with arbitrary number of parameters)
c = sphere([0.1, 0.2, 0.3, 0.4])
g = sphere_derivative([0.1, 0.2, 0.3, 0.4])
h = sphere_hessian([0.1, 0.2, 0.3, 0.4])

# Styblinski-Tang
c = styblinski_tang([0.1, 0.2])
g = styblinski_tang_derivative([0.1, 0.2])
h = styblinski_tang_hessian([0.1, 0.2])

# Three-hump-camel
c = threehumpcamel([0.1, 0.2])
g = threehumpcamel_derivative([0.1, 0.2])
h = threehumpcamel_hessian([0.1, 0.2])
```


## License

Licensed under either of

 - Apache License, Version 2.0, ([LICENSE-APACHE](https://github.com/argmin-rs/argmin/blob/main/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
 - MIT License ([LICENSE-MIT](https://github.com/argmin-rs/argmin/blob/main/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
