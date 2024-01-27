+++
title = "argmin 0.7.0 and argmin-math 0.2.0 released"
description = ""
date = 2022-08-28T00:00:00+00:00
updated = 2022-08-28T00:00:00+00:00
draft = false
template = "blog/page.html"

[taxonomies]
authors = ["Stefan Kroboth"]

[extra]
+++

> <b>argmin</b> offers a range of numerical optimization methods in Rust.

In the previous release I unfortunately accidentally removed the possibility to turn off building without the `ndarray-linalg` dependency when using the `ndarray` backend in argmin-math.
`ndarray-linalg` is essentially only needed for the matrix inverse, which only a few solvers require.
However, it links against a BLAS which can cause problems that are particularly frustrating when it is not even needed.

In this release, this possibility is added again.
Each `ndarray`-related feature now also comes with a `*-nolinalg*` version which turns off building `ndarray-linalg`.
Note that this will not implement `ArgminInv` for the `ndarray` backend and therefore certain solvers (for instance the Newton method) will not work with `ndarray` types.
More details can be found in the [argmin-math docs](https://docs.rs/argmin-math/). 

I was unsure if this fix was a breaking change because of the way argmin depends on argmin-math, therefore I decided to go ahead and release version 0.7 of argmin and version 0.2 of argmin-math just to be safe.

This also gave me the chance to release other (some of them definitely breaking) changes which were made recently.

The change with the most impact was made by [@vbkaisetsu](https://github.com/vbkaisetsu) (Thanks!) who added [L1 regularization](L1-regularization) to L-BFGS (also known as OWL-QN). 
This method can be enabled with the `with_l1_regularization` method of `LBFGS` which takes the non-negative L1 coefficient as input.

```rust
let solver = LBFGS::new(linesearch, 7).with_l1_regularization(1.0)?;
```

[@vbkaisetsu](https://github.com/vbkaisetsu) also provided an [example](https://github.com/argmin-rs/argmin/tree/argmin-v0.7.0/argmin/examples/owl_qn.rs). 

A minor downside of this addition is that the number of trait bounds on the types used in L-BFGS increased.
If you are using one of the default backends, this is unlikely to affect you though.

The OWL-QN implementation required changes and additions to `argmin-math`, for instance the traits `ArgminSignum` and `ArgminL1Norm`. 

To make the already existing `ArgminNorm` trait (which computes the L2 norm) more consistent with `ArgminL1Norm`, I renamed it to `ArgminL2Norm`.

And while they were at it, [@vbkaisetsu](https://github.com/vbkaisetsu) also fixed wrong type alias names in some of the examples.

Shortly after releasing the previous version [@renato145](https://github.com/renato145) reported [a bug](https://github.com/argmin-rs/argmin/issues/246) in the `Dogleg` method which I fixed.
Thanks again for reporting!


Overall, updating argmin from 0.6 to 0.7 should be fairly straight forward in most cases.


<br>
<script async defer src="https://buttons.github.io/buttons.js"></script>
<p align="center">
<a class="github-button" href="https://github.com/argmin-rs/argmin" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star argmin-rs/argmin on GitHub">Star</a>
<a class="github-button" href="https://github.com/argmin-rs/argmin/subscription" data-icon="octicon-eye" data-size="large" data-show-count="true" aria-label="Watch argmin-rs/argmin on GitHub">Watch</a>
<a class="github-button" href="https://github.com/argmin-rs/argmin/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork argmin-rs/argmin on GitHub">Fork</a>
<a class="github-button" href="https://github.com/sponsors/stefan-k" data-icon="octicon-heart" data-size="large" aria-label="Sponsor @stefan-k on GitHub">Sponsor</a>
</p>
