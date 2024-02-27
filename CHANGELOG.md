# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (since argmin version 0.6.0),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (since argmin version 0.5.0).

## [argmin unreleased]
* Added a new GUI observer called Spectator (`argmin-observer-spectator` and `spectator` packages) (@stefan-k, #311)

## [argmin-math unreleased]

## [argmin v0.10.0] 2024-02-27

### Added
* Added optional timeout to Executor (@stefan-k, #405)

### Changed
* Users can now set ParticleSwarms random number generator (@jonboh, #383)
* Moved all observers and checkpointing into dedicated crates. This led to substantially fewer dependencies for argmin itself
  - `SlogLogger` moved into `argmin-observer-slog` (@stefan-k, #311)
  - `WriteToFile` was renamed to `ParamWriter` and moved into `argmin-observer-paramwriter` (@stefan-k, #395)
  - `FileCheckpoint` moved into `argmin-checkpointing-file` (@stefan-k, #311)
* Added `state` to `observe_init` of `Observe` trait (@stefan-k, #311)
* All crates are now in the `crates` directory (@stefan-k, #415)
* All examples are now crates and are in the `examples` directory. (@stefan-k, #387)
* Removed `SerializeAlias` and `DeserializeAlias` traits since they are not necessary anymore (@stefan-k, #395)
* Residuals in CG are now handled by `IterState` (@stefan-k, #408)
* Interrupt handling now includes `SIGINT`, `SIGTERM` and `SIGHUP` and `KeyboardInterrupt` was renamed to `Interrupt` (@stefan-k, #413)
* Error handling in `LBFGS` was slightly improved (@stefan-k, #416)
* `KvValue::get_float(&self)` now works for all kinds of `KvValue` (@stefan-k, #311)
* Updated gnuplot from 0.0.39 to 0.0.4 (#394)

### Fixed
* Fixed residuals handling in `GaussNewton` (@gmilleramilar, @stefan-k, #343, #392)

## [argmin-math v0.4.0] 2024-02-27

### Added

* Implemented ArgminInv for f32/f64 (1D) (@sdrap, #346)
* Test cases for ArgminMinMax were added (@Shreyan11, #391)

### Changed

* Default linalg backend for development is now MKL, which makes development on Windows possible. This required all ndarray tests to be moved in dedicated crates (@Tastaturtaste, #369)
* `ArgminRandom::rand_from_range(..)` now also takes a random number generator. This allows for setting the seed manually (@jonboh, #383)

### Removed

* All `*-serde` features were removed, as they are not necessary anymore. (@stefan-k, #387)

## [argmin-observer-slog v0.1.0] 2024-02-27

Initial release.

## [argmin-observer-paramwriter v0.1.0] 2024-02-27

Initial release.

## [argmin-checkpointing-file v0.1.0] 2024-02-27

Initial release.

## [argmin-testfunctions-py v0.0.1] 2024-02-16

* The first version of a Python wrapper around `argmin_testfunctions` was released. 
* The fact that this is still experimental is expressed with the version number 0.0.1. Eventually this module's version may end up in lockstep with `argmin_testfunctions`' version number.

## [argmin_testfunctions v0.2.0] 2024-02-16

All work of this release was done by @stefan-k.

### Added

* Added derivative and Hessian calculation to all test functions.
* Tested all derivatives and Hessian against finite differences (with `finitediff`) on multiple positions with `proptest`.
* Added const generics versions for the derivatives and Hessians of multi dimensional test functions to avoid allocations.
* Those test functions which operate on a fixed number of parameters now take a fixed length array as input instead of a slice.

### Fixed

* Bugs in the Levy test function were fixed.

### Changed

* Rust edition 2021
* Repository was moved into argmin monorepo
* `rosenbrock` now only takes the parameters as input. For setting the additional optional parameters `a` and `b` one can use `rosenbrock_ab`
* All `rosenbrock_2d*` functions were removed.

## [argmin v0.9.0] 2024-01-06

### Added

* Added a simple example of Nelder-Mead usage (@cjordan, #359)

### Fixed

* Clippy lints (@jjbayer, @stefan-k, #341, #376, #380)
* Fixed wrong example link (@itrumper, #352)
* Correct line search to search over gradient (@DevonMorris, #354)
* SteepestDescent now correctly keeps track of prev_param (@DevonMorris, #362)
* Added a missing feature for testing argmin-math in the docs (@stefan-k, #371)
* Fixed a typo in argmin book (@imeckler, #373)
* Fixed crate versions in book (@stefan-k, #375)

### Changed

* Updated gnuplot to 0.0.39 (@stefan-k, #364)
* Switched to resolver = 2 for entire workspace (@stefan-k, #372)

## [argmin v0.8.1] 2023-02-20

### Added

* Mentioned `cobyla` in docs and README (@relf)

### Fixed

* Restored Serialize/Deserialize derives for ObserverMode (@maoe)

### Changed

* Updated ctrlc to 3.2.4 (@stefan-k)
* Updated rayon to 1.6 (@stefan-k)
* Updated slog to 2.7 (@stefan-k)
* Updated slog-term to 2.9 (@stefan-k)
* Updated slog-json to 2.6 (@stefan-k)

## [argmin v0.8.0] 2023-01-28

### Added

* Added a `full` feature which enables all features (@stefan-k)
* Added a particle swarm optimization example using the `nalgebra` math backend (@stefan-k)
* Added a L-BFGS example using the `nalgebra` math backend (@stefan-k)

### Changed

* Improved termination handling (@relf, #305)
  - Introduced `TerminationStatus` which can either be `Terminated(TerminationReason)` or `NotTerminated`
  - The fields of the `TerminationReason` enum were reduced to cases potentially relevant to any solver. Solver-specific cases can be handled with `SolverExit(String)`.
  - Added new `TerminationReason::KeyboardInterrupt` and `TerminationReason::Converged`
* Changes to KV store used to get values from the solver to observers (@stefan-k)
  - KV store is  now typed (via `KvValue`), which means that observers can now use the actual numbers (in the past those values were only `dyn Display` and as such could only be printed)
  - `make_kv!` renamed to `kv!`
* Solver does not need to be `Serialize` anymore when `serde1` feature is enabled. This was an oversight reported by @relf and fixed by @stefan-k
* Better calculation of elapsed time inside `Executor` (@TheIronBorn)
* The check whether the target cost is reached is now based on the current best cost function value rather than the current cost function value (@relf)


## [argmin-math v0.3.0] 2023-01-28

### Added

* Added support for `nalgebra` 0.32

### Changed

* All math backends now implement all math related traits (apart from the `Vec` backend which does not implement `ArgminInv`). Therefore (almost) all solvers work with all backends. (@hypotrochoid, @stefan-k)
* More primitive and complex types are now covered as well (@stefan-k)
* Reached 100% test coverage (@stefan-k)
* `InverseError` is not `Clone` anymore (@stefan-k)
* Upgraded `ndarray-linalg` to version 0.16 for the `ndarray` 0.15 backend


## [argmin-math v0.2.1] 2022-09-09

### Added
- Added ArgminMinMax and ArgminSignum implementation for nalgebra types. Now the L-BFGS should work work with the nalgebra backend again. (#257, #258, @stefan-k)
- Added an L-BFGS example using the nalgebra backend (#258, @stefan-k)

## [argmin v0.7.0] and [argmin-math v0.2.0] 2022-08-28

### Added
- L1 regularization added to L-BFGS (also known as OWL-QN). L-BFGS now has more trait bounds, please consult the documentation if this causes problems. (#244, #248, #250, Thanks to @vbkaisetsu)

### Changed
- The `ArgminL1Norm` trait was added by @vbkaisetsu for the L1 regularization in L-BFGS. In order to make the L2 norm more consistent, the corresponding trait was renamed from `ArgminNorm` to `ArgminL2Norm` and the `norm` method was renamed to `l2_norm`. (#253, @stefan-k)

### Fixed
- Version 0.6 accidentally removed the possibility to deactivate the `ndarray-linalg` dependency in `argmin-math`. This leads to problems in downstream crates which do not need the functionality of `ndarray-linalg` and want to avoid linking angainst a BLAS. In this version, this functionality was added again. Please consult the documentation of `argmin-math` for details on the available backends and their configuration. (#243, #249, @stefan-k)
- Fixed type alias names in examples (#245, Thanks to @vbkaisetsu)
- Fixed a bug in dogleg method (#246, #247, @stefan-k, Thanks to @renato145 for reporting!)

## [argmin v0.6.0] and [argmin-math v0.1.0] 2022-08-09

This is a rather large release with many (breaking) changes.

- Code related to the abstraction of math backends was split of into the `argmin-math` crate. This crate offers now the traits related to math operations as well as their implementations for various backends (vec, ndarray, nalgebra). The backends can be turned on and off as needed. It also supports different versions of the backends.
- The `serde` dependency is now optional and feature-gated via `serde1`. Disabling the `serde1` feature disables checkpointing and some logging.
- The `ArgminOp` trait was removed and replaced with more specialized traits such as `Operator`, `CostFunction`, `Gradient`, `Jacobian`, `Hessian`, `Anneal`. 
  All of the above mentioned traits come with `bulk_*` methods which allow one to compute cost function, gradients,... in parallel as long as the `rayon` feature is enabled. This is so far used in Particle Swarm Optimization.
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
- remove unnecessary Default bound on NelderMead (#73, @vadixidav)
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

- CG improvements (@stefan-k)

## older versions

For older versions please see the Git history.

[argmin unreleased]: https://github.com/argmin-rs/argmin/compare/argmin-v0.9.0...HEAD
[argmin-math unreleased]: https://github.com/argmin-rs/argmin/compare/argmin-math-v0.3.0...HEAD
[argmin_testfunctions unreleased]: https://github.com/argmin-rs/argmin/compare/argmin-v0.9.0...HEAD
[argmin v0.10.0]: https://github.com/argmin-rs/argmin/compare/argmin-v0.9.0...argmin-v0.10.0
[argmin v0.9.0]: https://github.com/argmin-rs/argmin/compare/argmin-v0.8.1...argmin-v0.9.0
[argmin v0.8.1]: https://github.com/argmin-rs/argmin/compare/argmin-v0.8.0...argmin-v0.8.1
[argmin v0.8.0]: https://github.com/argmin-rs/argmin/compare/argmin-v0.7.0...argmin-v0.8.0
[argmin v0.7.0]: https://github.com/argmin-rs/argmin/compare/argmin-v0.6.0...argmin-v0.7.0
[argmin v0.6.0]: https://github.com/argmin-rs/argmin/compare/v0.5.1...argmin-v0.6.0
[argmin-math v0.4.0]: https://github.com/argmin-rs/argmin/compare/argmin-math-v0.3.0...argmin-math-v0.4.0
[argmin-math v0.3.0]: https://github.com/argmin-rs/argmin/compare/argmin-math-v0.2.1...argmin-math-v0.3.0
[argmin-math v0.2.1]: https://github.com/argmin-rs/argmin/compare/argmin-math-v0.2.0...argmin-math-v0.2.1
[argmin-math v0.2.0]: https://github.com/argmin-rs/argmin/compare/argmin-math-v0.1.0...argmin-math-v0.2.0
[argmin-math v0.1.0]: https://github.com/argmin-rs/argmin/compare/v0.5.1...argmin-math-v0.1.0
[argmin-testfunctions-py v0.0.1]: https://github.com/argmin-rs/argmin/compare/argmin-v0.9.0...argmin-testfunctions-py-v0.0.1
[argmin_testfunctions v0.2.0]: https://github.com/argmin-rs/argmin/compare/argmin-v0.9.0...argmin_testfunctions-v0.2.0
[argmin-observer-slog v0.1.0]: https://github.com/argmin-rs/argmin/compare/v0.9.0...argmin-observer-slog-v0.1.0
[argmin-observer-paramwriter v0.1.0]: https://github.com/argmin-rs/argmin/compare/v0.9.0...argmin-observer-paramwriter-v0.1.0
[argmin-checkpointing-file v0.1.0]: https://github.com/argmin-rs/argmin/compare/v0.9.0...argmin-checkpointing-file-v0.1.0
