# Contributing

This crate is looking for contributors!
Potential projects can be found in the [Github issues](https://github.com/argmin-rs/argmin/issues), but feel free to suggest your own ideas.
Besides adding optimization methods and new features, other contributions are also highly welcome, for instance improving performance, documentation, writing examples (with real world problems), developing tests, adding observers, implementing a C interface or
[Python wrappers](https://github.com/argmin-rs/pyargmin).
Bug reports (and fixes) are of course also highly appreciated.

## Running the tests

The repository is organized as a workspace with the two crates `argmin` and `argmin-math`.
It is recommended to run the tests for both crates individually, as this simplifies the choice of features.
In case of `argmin`, all combinations of features should lead to working tests.
For `argmin-math`, this is not the case, since some of the features are mutually exclusive.

Therefore it is recommended to run the tests for `argmin` from the root directory of the repository like this:

```bash
cargo test -p argmin --all-features
```

This will work for `--all-features` and any other combination of features.
Note that not all test will run if only a subset of the features is enabled.

In terms of `argmin-math`, one can just test the default features:

```bash
cargo test -p argmin-math
```

Or the default features plus the latest `ndarray`/`nalgebra` backends:

```bash
cargo test -p argmin-math --features "latest_all"
```

Individual backends can be tested as well; however, care has to be taken to not add two backends of the same kind but with different versions, as that may not work.
