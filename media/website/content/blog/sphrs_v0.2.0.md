+++
title = "sphrs 0.2.0 released"
description = ""
date = 2023-01-15T00:00:00+00:00
updated = 2023-01-15T00:00:00+00:00
draft = false
template = "blog/page.html"

[taxonomies]
authors = ["Stefan Kroboth"]

[extra]
math = true
+++

This post announces the release of version 0.2.0 of <b>sphrs</b>, a spherical harmonics and solid harmonics library for Rust.

## Introduction to sphrs

The crate computes

* real and complex spherical harmonics
* real and complex regular solid harmonics
* real and complex irregular solid harmonics

using Cartesian and polar coordinates. 

Here is a small example to illustrate the API which computes the real valued spherical harmonic of degree \\(l=2\\) and order \\(m=1\\) at Cartesian position \\((x, y, z) = (1.0, 0.2, 1.4)\\):

```rust
use sphrs::{Coordinates, RealSH, SHEval};

// l = 2
let degree = 2;
// m = 1
let order = 1;

// Define the position where the SH will be evaluated at
// in Cartesian coordinates
let p = Coordinates::cartesian(1.0, 0.2, 1.4);

// Compute the real-valued SH value at `p` for l = 2, m = 1
let computed_sh = RealSH::Spherical.eval(degree, order, &p);

println!("SH ({}, {}): {:?}", degree, order, computed_sh);
```

The library also provides the type `HarmonicsSet` which computes all harmonics up to a given order and returns them in a `Vec`:

```rust
use sphrs::{ComplexSH, Coordinates, HarmonicsSet};

// l = 3
let degree = 3;

// Create the harmonics set (in this case for complex SH)
let sh = HarmonicsSet::new(degree, ComplexSH::Spherical);

// Position in spherical coordinates where the set is evaluated at
let p = Coordinates::spherical(1.0, 0.8, 0.4);

// Evaluate. Returns a `Vec<f64>`
let set = sh.eval(&p);

println!("SH up to degree {}: {:?}", degree, set);

// The individual SHs can also be multiplied with coefficients
// Note: `coeff` must have the same length as the number of SH.
let coeff = vec![2.0; sh.num_sh()];
let set = sh.eval_with_coefficients(&p, coeff.as_slice());

println!("SH up to degree {}: {:?}", degree, set);
```

More details on how to use sphrs can be found in the [API documentation](https://docs.rs/sphrs/latest/sphrs/). 

Harmonics up to third order are implemented according to their respective formulas in order to achieve good performance. 
For higher orders, a recursive implementation is used.

## Version 0.2.0

The new version follows more than a year after the previous version 0.1.3 and comes with a couple of changes.
Most of those changes concern the project structure and the CI; however some are user-facing and breaking changes:

* Updated dependencies `num` (v0.4), `num-complex` (v0.4) and `num-traits` (v0.2)
* Updated to Rust Edition 2021
* Improved documentation
* Every `SH` in function names was changed to `sh`
* `RealSHType` and `ComplexSHType` lost their `Type` prefix and are now just `RealSH` and `ComplexSH`
* Functions which accepted `&dyn SHCoordinates` now accept `&impl SHCoordinates` instead
* The modules `coords` and `sh` are not `pub` anymore, all relevant types are re-exported at the crate root
* Removed automatic conversion of inputs to the type of the output when creating `Coordinates`, since this required users to provide the desired output type as type hints:

```rust
// Version 0.1.3 (required a type hint)
// Still compiles in 0.2.0
let p: Coordinates<f64> = Coordinates::spherical(1.0, 0.8, 0.4);

// Version 0.2.0 (does not require a type hint)
// The `T` of `Coordinates<T>` will be equal to the type of the input arguments
let p = Coordinates::spherical(1.0, 0.8, 0.4);
```

* The generics of `HarmonicsSet` were also changed such that type hints are not necessary anymore. In fact, the type hints needed with 0.1.3 will break compilation with 0.2.0:

```rust
// Version 0.1.3 (required a type hint)
// This will *not* compile in 0.2.0
let sh: HarmonicsSet<f64, _, _> = HarmonicsSet::new(degree, RealSHType::Spherical);

// Version 0.2.0 (does not require a type hint)
let sh = HarmonicsSet::new(degree, RealSHType::Spherical);
```

## Acknowledgements

The implementation is inspired by Google's [spherical-harmonics](https://github.com/google/spherical-harmonics) library and follows the math outlined in [this document](https://basesandframes.files.wordpress.com/2016/05/spherical_harmonic_lighting_gritty_details_green_2003.pdf).

## Future steps

In the future this crate will hopefully see an increase in test coverage and potentially a Python wrapper. 
If you have any ideas for further development or if you even want to contribute, feel free to [open an issue](https://github.com/argmin-rs/sphrs/issues).


<br>
<script async defer src="https://buttons.github.io/buttons.js"></script>
<p align="center">
<a class="github-button" href="https://github.com/argmin-rs/sphrs" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star argmin-rs/sphrs on GitHub">Star</a>
<a class="github-button" href="https://github.com/argmin-rs/sphrs/subscription" data-icon="octicon-eye" data-size="large" data-show-count="true" aria-label="Watch argmin-rs/sphrs on GitHub">Watch</a>
<a class="github-button" href="https://github.com/argmin-rs/sphrs/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork argmin-rs/sphrs on GitHub">Fork</a>
<a class="github-button" href="https://github.com/sponsors/stefan-k" data-icon="octicon-heart" data-size="large" aria-label="Sponsor @stefan-k on GitHub">Sponsor</a>
</p>
