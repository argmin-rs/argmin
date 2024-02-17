<p align="center">
  <img
    width="400"
    src="https://raw.githubusercontent.com/argmin-rs/argmin/main/media/logo.png"
  />
</p>
<h1 align="center">Spectator</h1>

<p align="center">
  <a href="https://argmin-rs.org">Website</a>
  |
  <a href="https://argmin-rs.org/book/">Book</a>
  |
  <a href="https://docs.rs/spectator">Docs (latest release)</a>
  |
  <a href="https://argmin-rs.github.io/argmin/spectator/index.html">Docs (main branch)</a>
</p>

<p align="center">
  <a href="https://crates.io/crates/spectator"
    ><img
      src="https://img.shields.io/crates/v/spectator?style=flat-square"
      alt="Crates.io version"
  /></a>
  <a href="https://crates.io/crates/spectator"
    ><img
      src="https://img.shields.io/crates/d/spectator?style=flat-square"
      alt="Crates.io downloads"
  /></a>
  <a href="https://github.com/argmin-rs/argmin/actions"
    ><img
      src="https://img.shields.io/github/actions/workflow/status/argmin-rs/argmin/ci.yml?branch=main&label=argmin CI&style=flat-square"
      alt="GitHub Actions workflow status"
  /></a>
  <img
    src="https://img.shields.io/crates/l/spectator?style=flat-square"
    alt="License"
  />
  <a href="https://discord.gg/fYB8AwxxMW"
    ><img
      src="https://img.shields.io/discord/1189119565335109683?style=flat-square&label=argmin%20Discord"
      alt="argmin Discord"
  /></a>
</p>

Spectator is a GUI tool to observe the progress of argmin optimization runs. 

## Installation

The preferred way is to install directly from crates.io:

```bash
cargo install spectator --locked
```

Alternatively, one can clone the repo and install/run from there:

```bash
git clone https://github.com/argmin-rs/argmin.git
cd argmin

# Compile and run from the repo...
cargo build -p spectator --release
./target/release/spectator

# .. or directly run from the repo...
cargo run -p spectator --release

# ... or install locally
cargo install -p spectator
spectator
```

## Usage

```bash
spectator --host 127.0.0.1 --port 5498
```

The optional options `--host` and `--port` indicate the host and port spectator binds to.
By default, spectator will bind to `0.0.0.0:5498`.

The argmin optimization run which should be observed needs to use the spectator observer which 
can be found [in the `argmin-observer-spectator` crate](https://crates.io/crates/argmin-observer-spectator).

## License

Licensed under either of

  * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
  * MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you,
as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
