<p align="center">
  <img
    width="400"
    src="https://raw.githubusercontent.com/argmin-rs/argmin/main/media/logo.png"
  />
</p>
<h1 align="center">argmin_testfunctions</h1>

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
  <a href="https://crates.io/crates/argmin"
    ><img
      src="https://img.shields.io/crates/v/argmin_testfunctions?style=flat-square"
      alt="Crates.io version"
  /></a>
  <a href="https://crates.io/crates/argmin"
    ><img
      src="https://img.shields.io/crates/d/argmin_testfunctions?style=flat-square"
      alt="Crates.io downloads"
  /></a>
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

A collection of two- and multidimensional test functions (and their derivatives and Hessians) for optimization algorithms. 
For two-dimensional test functions, the derivate and Hessian calculation does not allocate. For multi-dimensional tes functions,
the derivative and Hessian calculation comes in two variants. One variant returns `Vec`s and hence does allocate. This is 
needed for cases, where the number of parameters is only known at run time. In case the number of parameters are known at
compile-time, the `_const` variants can be used, which return fixed size arrays and hence do not allocate.

The derivative and Hessian calculation is always named `<test function name>_derivative` and
`<test function name>_hessian`, respectively. The const generics variants are defined as
`<test function name>_derivative_const` and `<test function name>_hessian_const`.

Some functions, such as `ackley`, `rosenbrock` and `rastrigin` come with additional optional parameters which change
the shape of the functions. These additional parameters are exposed in `ackley_abc`, `rosenbrock_ab` and `rastrigin_a`. 

All functions are generic over their inputs and work with `[f64]` and `[f32]`.

For a list of all implemented functions see the documentation linked above.

## Python wrapper

Thanks to the python module [`argmin-testfunctions-py`](https://pypi.org/project/argmin-testfunctions-py/), it is possible to use the functions in Python as well.
Note that the derivative and Hessian calculation used in the wrapper will always allocate.

## Running the tests and benchmarks

The tests can be run with

```bash
cargo test
```

The test functions derivatives and Hessians are tested against [finitediff](https://crates.io/crates/finitediff) using
[proptest](https://crates.io/crates/proptest) to sample the functions at various points. 

All functions are benchmarked using [criterion.rs](https://crates.io/crates/criterion). Run the benchmarks with

```bash
cargo bench
```

The report is available in `target/criterion/report/index.html`.

## Contributing

This library is the most useful the more test functions it contains, therefore any contributions are highly
welcome. For inspiration on what to implement and how to proceed, feel free to have a look at this
[issue](https://github.com/argmin-rs/argmin/issues/450).

While most of the implemented functions are probably already quite efficient, there are probably a few
which may profit from performance improvements.

## License

Licensed under either of

  * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
  * MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you,
as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
