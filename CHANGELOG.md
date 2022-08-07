# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (since argmin version 0.6.0),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (since argmin version 0.5.0).

## argmin [Unreleased]

## argmin-math [Unreleased]

## [argmin v0.6.0] and [argmin-math v0.1.0] 2022-08-09

This is a rather large release with many (breaking) changes.

- Code related to the abstraction of math backends was split of into the `argmin-math` crate. This crate offers now the traits related to math operations as well as their implementations for various backends (vec, ndarray, nalgebra). The backends can be turned on and off as needed. It also supports different versions of the backends.
- The `serde` depenency is now optional and feature-gated via `serde1`. Disabling the `serde1` feature disables checkpointing and some logging.
- The `ArgminOp` trait was removed and replaced with more specialized traits such as `Operator`, `CostFunction`, `Gradient`, `Jacobian`, `Hessian`, `Anneal`. 
  All of the above mentioned traits come with `bulk_*` methods wich allow one to compute cost function, gradients,... in parallel as long as the `rayon` feature is enabled. This is so far used in Particle Swarm Optimization.
- The handling of solver-internal state was redesigned completely to allow for various kinds of state, depending on the solver. There is a `State` trait now which defines the basic functionality that a state must offer. 
- Many aspects were redesigned to avoid unnecessary cloning
- `ArgminResult` was renamed to `OptimizationResult` and also has a more useful API now. It returns the solver, the problem and the final internal state.
- The `Solver` trait was redesigned 
- The solver is now configured/initialized via the `configure` method on `Executor`
- KV does not store Strings anymore but rather `dyn Display`. This avoids unnecessarily stringifying variables when no observer is used.
- Checkpointing is now a trait (so everyone can implement their own checkpointing mechanisms). Also the handling of checkpoints was completely redesigned.
- Made the API surface a bit more consistent. 
- The prelude was removed
- `OpWrapper` was renamed to `Problem` and has now a more consistent interface. It still keeps track of the function evaluations, but to be flexible enough to support future or user-defined traits, these are stored in a `HashMap` now.
- PSO was redesigned and a `PopulationState` was introduced which replaces `IterState` in PSO
- Unnecessary trait bounds on solvers were removed
- Removed many `unwrap`s and instead return errors with meaningful error messages (for instance when something isn't initialized properly)
- The optional `stdweb` dependency was removed
- Documentation was improved. The README is now rather short and not created from the `lib.rs` file anymore. 
- The website was updated and the "argmin book" was published, which is so far mainly a collection of short tutorials but will be extended in the future.
- A lot more tests and doc tests were added
- `ArgminError` is now `non_exhaustive`.
- Added support for nalgebra 0.31
- Many now useless macros were removed
- Improved the CI to better test all aspects of the code
- Renamed `termination_reaosn` method of `State` to `terminate_with`
- Renamed all `*grad*` methods of `IterState` to `*gradient*`
- Code of conduct was added
- Brent's optimization method was added (#77, thanks to @Armavica)
- Fixed a bug in Nelder-Mead which was reported by @bivhitscar (thanks!)
- Remove the Default trait bound on TrustRegion and Steihaug impls (#192, thanks to @cfunky)

## argmin v0.5.1 (16 February 2022)

- Fixed Bug in HagerZhang line search (#2, #184, @wfsteiner)
- Removed Default trait bounds on TrustRegion and Steihaug implementations (#187, #192, @cfunky)
- Inverse Hessians are now part of IterState, therefore the final inverse Hessian can be retrieved after an optimization run (#185, #186, @stefan-k)

## argmin v0.5.0 (10 January 2022)

- Faster CI pipeline (#179, @stefan-k)
- Removed CircleCI and added rustfmt check to Github Actions (#178, @stefan-k)
- Automatically build documentation in CI when merging a PR into main (#149, #176, @stefan-k) 
- Added a section to documentation where to find examples for current release and main branch, removed other links (#145, #174, @stefan-k)
- Fixed warnings when building docs and added building docs to the CI (#173, @stefan-k)
- The required features for each example are now indicated in Cargo.toml (#171, #147, @stefan-k)
- CI now includes compiling to various WASM targets (#89, #170, @stefan-k)
- Branch master renamed to main (#148, @stefan-k)
- nalgebra updated from 0.29.0 to 0.30.0 (#169)
- WASM features now mentioned in documentation and README.md (#167, @Glitchy-Tozier)
- Added tests for backtracking linesearch (#168, @stefan-k)
- Removed unsafe code from the vec-based math module (#166, @stefan-k)
- Added tests for GaussNewton method and fixed GaussNewton example (#164, @stefan-k)
- Added tests for Newton method (#163, @stefan-k)
- Treat a new parameter as "best" when both current and previous cost function values are Inf (#162, @stefan-k)
- Corrected documentation of PSO and removed an unnecessary trait bound on Hessian (#161, #141, @stefan-k, @TheIronBorn)
- Moved to edition 2021 (#160, @stefan-k)
- SA acceptance now based on current cost, not previous (fix) (#157, #159, @stefan-k, @TheIronBorn)
- LineSearchCondition now uses references (#158, @w1th0utnam3)
- Counting of sub problem function counts fixed (#154, #156, @stefan-k, @w1th0utnam3)
- Fixed incorrect checking for new best solution (#151, #152, @stefan-k, @Glitchy-Tozier)
- Fixed simulated annealing always accepting the first iteration (#150, #153, @dariogoetz)
- Fixed inconsistency between state and alpha value in Backtracking linesearch (#155, @w1th0utnam3)
- Improved clippy linting in CI (#146, @stefan-k)
- Unnecessary semi-colon in macro removed (#143, @CattleProdigy)
- Allow any RNG in SA and improve example (#139, @TheIronBorn)
- Make use of slog a feature, improve tests (#136, @ThatGeoGuy)

## argmin v0.4.7 (14 August 2021)

- Moved to Github actions (#130, @stefan-k)
- Updated README.md (#131, @stefan-k)
- Updated nalgebra from 0.28 to 0.29 (#133)

## argmin v0.4.6 (18 July 2021)

- updated dependencies (#121, #123, #129, @stefan-k):
  + ndarray 0.15
  + ndarray-linalg 0.14
  + appox 0.5
  + nalgebra 0.28
  + ndarray-rand 0.14
  + num-complex 0.4
  + finitediff 0.1.4

## argmin v0.4.5 

- Squash warnings for Nalgebra 0.26.x (#118, #117, @CattleProdigy)

## argmin v0.4.4 

- Finally started writing a changelog.
- Performance improvements (#111, #112, @sdd)

## argmin v0.4.3

- Downgraded argmin-rand to 0.13 to match ndarray 0.14

## argmin v0.4.2

- Fix lazy evaluation of gradients in line searches (#101, @w1th0utnam3)
- Various updated dependencies

## argmin v0.4.1

- Typo

## argmin 0.4.0

- nalgebra support (#68, @Maher4Ever)
- remove unecessary Default bound on NelderMead (#73, @vadixidav)
- Various updated dependencies

## argmin 0.3.1

- remove finitediff from ndarrayl feature (#61, @optozorax)
- MoreThuente: Added error check for NaN or Inf (#57, @MattBurn)

## argmin 0.3.0

- Golden-section search (#49, @nilgoyette)
- Allow users to choose floating point precision (#50, @stefan-k)
- Remove Clone trait bound from ArgminOp (#48, @stefan-k)
- Remove Serialize trait bound on ArgminOp (#36, @stefan-k)
- Moved from failure to anyhow and thiserror (#44, @stefan-k)
- Added easier access to op and state of ArgminResult (#45, @stefan-k)
- No reexport of argmin_testfunctions (#46, @stefan-k)
- Exposed stopping criterion tolerances of Quasi-Newton methods to user(#43, @stefan-k)
- Exposed stopping criterion tolerance of NewtonCG method to user (@stefan-k)
- Exposed stopping criterion tolerances of Gauss Newton methods to user (@stefan-k)
- Exposed L-BFGS stopping criterion tolerances to user (#37, @stefan-k)
- Removed need for unwrap in cstep MoreThuente LineSearch (#38, @MattBurn)
- Removed Send and Sync trait bounds from ArgminOp (#33, @stefan-k)

## argmin 0.2.6

- Brent's method (#22, @xemwebe)

## argmin 0.2.5

- Particle Swarm Optimization (@jjbayer)
- Derive Clone trait (#14, @rth)
- Test convergence (#13, @rth)
- Lints (#11, @rth)
- Improvements in CG method (@stefan-k)

## argmin 0.2.4

- CG improvments (@stefan-k)

## older versions

For older versions please see the Git history.

[Unreleased]: https://github.com/argmin-rs/argmin/compare/v0.5.1...HEAD
[argmin v0.6.0]: https://github.com/argmin-rs/argmin/compare/v0.5.1...argmin_v0.6.0
[argmin-math v0.1.0]: https://github.com/argmin-rs/argmin/compare/v0.5.1...argmin_v0.6.0
