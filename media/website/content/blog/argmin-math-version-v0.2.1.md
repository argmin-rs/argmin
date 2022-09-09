+++
title = "argmin-math 0.2.1 released"
description = ""
date = 2022-09-09T00:00:00+00:00
updated = 2022-09-09T00:00:00+00:00
draft = false
template = "blog/page.html"

[taxonomies]
authors = ["Stefan Kroboth"]

[extra]
+++

> <b>argmin</b> offers a range of numerical optimization methods in Rust.

The previous release added (optional) L1-regularization to L-BFGS which increased the trait bounds on the acceptable types.
Unfortunately these traits where only partially implemented for the `nalgebra` backend, therefore L-BFGS could not be used with `nalgebra` types anymore.
The 0.2.1 release of `argmin-math` fixes this and adds an L-BFGS example for `nalgebra` types to the `examples/` directory.


<br>
<script async defer src="https://buttons.github.io/buttons.js"></script>
<p align="center">
<a class="github-button" href="https://github.com/argmin-rs/argmin" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star argmin-rs/argmin on GitHub">Star</a>
<a class="github-button" href="https://github.com/argmin-rs/argmin/subscription" data-icon="octicon-eye" data-size="large" data-show-count="true" aria-label="Watch argmin-rs/argmin on GitHub">Watch</a>
<a class="github-button" href="https://github.com/argmin-rs/argmin/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork argmin-rs/argmin on GitHub">Fork</a>
<a class="github-button" href="https://github.com/sponsors/stefan-k" data-icon="octicon-heart" data-size="large" aria-label="Sponsor @stefan-k on GitHub">Sponsor</a>
</p>
