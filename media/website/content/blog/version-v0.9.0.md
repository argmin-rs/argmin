+++
title = "argmin 0.9.0 released"
description = ""
date = 2024-01-06T00:00:00+00:00
updated = 2024-01-06T00:00:00+00:00
draft = false
template = "blog/page.html"

[taxonomies]
authors = ["Stefan Kroboth"]

[extra]
+++

<b>argmin</b> is a Rust library which offers a range of numerical optimization methods and a framework for 
developing optimization algorithms.  For details about the design of argmin and its features I suggest having a look at
[the website](https://argmin-rs.org),
[the book](https://argmin-rs.org/book),
[Github](https://github.com/argmin-rs/argmin),
[crates.io](https://crates.io/crates/argmin) and
[lib.rs](https://lib.rs/crates/argmin).

This is a short summary of the changes in argmin 0.9.0 (and 0.8.1 as this version didn't get its own blog post) compared to 0.8.0.
Feel free to reach out via [Github](https://github.com/argmin-rs/argmin) or the new [Discord server](https://discord.gg/fYB8AwxxMW) if you encounter any issues during the upgrade.

## argmin 0.9.0

* Line search now correctly searches over gradient instead of parameter vector ([@DevonMorris](https://github.com/DevonMorris)) (**Breaking change**)
* SteepestDescent now correctly keeps track of prev_param ([@DevonMorris](https://github.com/DevonMorris))
* `ArgminInv` is now also implemented for 1D matrices ([@sdrap](https://github.com/sdrap))
* [@cjordan](https://github.com/cjordan) added another [example](https://github.com/argmin-rs/argmin/blob/main/argmin/examples/neldermead-cubic.rs) of how to use Nelder-Mead
* Fixed a couple of typos and mistakes in the documentation ([@itrumper](https://github.com/itrumper), [@imeckler](https://github.com/imeckler), [@stefan-k](https://github.com/stefan-k))

## argmin 0.8.1

* The `Serialize` and `Deserialize` derives for `ObserverMode` somehow got lost. This was fixed by [@maoe](https://github.com/maoe)

## Other news

The [`cobyla`](https://crates.io/crates/cobyla) crate was made compatible with argmin (Thanks to [@relf](https://github.com/relf)).

## New Discord server

Since the old Gitter channel didn't get much use we now finally have a [Discord server](https://discord.gg/fYB8AwxxMW)! I'm looking forward to seeing many of you there!

## Thanks

Big thanks to everyone who contributed to these releases: 
[@cjordan](https://github.com/cjordan)
[@DevonMorris](https://github.com/DevonMorris)
[@imeckler](https://github.com/imeckler)
[@itrumper](https://github.com/itrumper)
[@jjbayer](https://github.com/jjbayer)
[@maoe](https://github.com/maoe)
[@relf](https://github.com/relf)
[@sdrap](https://github.com/sdrap)
[@stefan-k](https://github.com/stefan-k),
and everyone who opened and responded to issues and discussions!



<br>
<script async defer src="https://buttons.github.io/buttons.js"></script>
<p align="center">
<a class="github-button" href="https://github.com/argmin-rs/argmin" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star argmin-rs/argmin on GitHub">Star</a>
<a class="github-button" href="https://github.com/argmin-rs/argmin/subscription" data-icon="octicon-eye" data-size="large" data-show-count="true" aria-label="Watch argmin-rs/argmin on GitHub">Watch</a>
<a class="github-button" href="https://github.com/argmin-rs/argmin/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork argmin-rs/argmin on GitHub">Fork</a>
<a class="github-button" href="https://github.com/sponsors/stefan-k" data-icon="octicon-heart" data-size="large" aria-label="Sponsor @stefan-k on GitHub">Sponsor</a>
</p>
