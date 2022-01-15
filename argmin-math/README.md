# argmin-math ![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue) [![argmin-math on crates.io](https://img.shields.io/crates/v/argmin-math)](https://crates.io/crates/argmin-math) [![argmin-math on docs.rs](https://docs.rs/argmin-math/badge.svg)](https://docs.rs/argmin-math) [![Source Code Repository](https://img.shields.io/badge/Code-On%20github.com-blue)](https://github.com/argmin-rs/argmin) [![argmin-math on deps.rs](https://deps.rs/repo/github/argmin-rs/argmin/status.svg)](https://deps.rs/repo/github/argmin-rs/argmin)

argmin-math provides mathematics related abstractions needed in argmin. It supports implementations of these abstractions for basic `Vec`s and for `ndarray` and `nalgebra`. The traits can of course also be implemented for your own types to make them compatible with argmin.


## Usage

Add the following line to your dependencies list:


```toml
[dependencies]
argmin-math = "0.1.0"
```

This will activate the `primitives` and `vec` features. For other backends see the section below.


### Features

Support for the various backends can be switched on via features:

| Feature | Default | Backend |
| --- | --- | --- |
| `primitives` | yes | basic integer and floating point types |
| `vec` | yes | `Vec`s (basic functionality) |
| `ndarray_latest` | no | `ndarray` (latest supported version) |
| `ndarray_latest-serde` | no | `ndarray` (latest supported version) + serde support |
| `ndarray_v0_15` | no | `ndarray` (version 0.15) |
| `ndarray_v0_15-serde` | no | `ndarray` (version 0.15) + serde support |
| `ndarray_v0_14` | no | `ndarray` (version 0.14) |
| `ndarray_v0_14-serde` | no | `ndarray` (version 0.14) + serde support |
| `ndarray_v0_13` | no | `ndarray` (version 0.13) |
| `ndarray_v0_13-serde` | no | `ndarray` (version 0.13) + serde support |
| `nalgebra_latest` | no | `nalgebra` (latest supported version) |
| `nalgebra_latest-serde` | no | `nalgebra` (latest supported version) + serde support |
| `nalgebra_v0_30` | no | `nalgebra` (version 0.30) |
| `nalgebra_v0_30-serde` | no | `nalgebra` (version 0.30) + serde support |
| `nalgebra_v0_29` | no | `nalgebra` (version 0.29) |
| `nalgebra_v0_29-serde` | no | `nalgebra` (version 0.29) + serde support |

It is not possible to activate two versions of the same backend.

The features labelled `*_latest*` are an alias for the most recent supported version of the respective backend.

Note that `argmin` by default compiles with `serde` support. Therefore, unless `serde` is deliberately turned off in `argmin`, it is necessary to activiate the `serde` support in `argmin-math` as well.

The default features `primitives` and `vec` can be turned off in order to only compile the trait definitions. If another backend is chosen, they will automatically be turned on again.


#### Example

Activate support for the latest supported `ndarray` version:


```toml
[dependencies]
argmin-math = { version = "0.1.0", features = ["ndarray_latest-serde"] }
```


## Contributing

You found a bug? Your favourite backend is not supported? Feel free to open an issue or ideally submit a PR.


## License

Licensed under either of

 - Apache License, Version 2.0, ([LICENSE-APACHE][__link0] or <http://www.apache.org/licenses/LICENSE-2.0>)
 - MIT License ([LICENSE-MIT][__link2] or <http://opensource.org/licenses/MIT>)

at your option.


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.


 [__link0]: https://github.com/argmin-rs/argmin/blob/main/LICENSE-APACHE
 [__link2]: https://github.com/argmin-rs/argmin/blob/main/LICENSE-MIT
