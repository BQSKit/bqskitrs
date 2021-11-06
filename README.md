# Native code for BQSKit

`bqskitrs` is a Python package that implements circuit instantiation and simulation to accelerate [BQSKit](https://github.com/bqsKit/bqskit).

## Installing bqskitrs

You can often install `bqskitrs` via PyPi with `pip`:
```
pip install bqskitrs
```

this will use pre-built wheels. Sometimes wheels are not available, so you must build from source.

## Building bqskitrs

### Linux

First make sure the version of pip you have is at least 20.0.2.
You can check via `pip -V` and upgrade via `python3 -m pip install -U pip`.

Next, install the dependencies. On Ubuntu 20.04 this is:

```bash
sudo apt install libopenblas-dev libceres-dev libgfortran-9-dev libeigen3-dev gfortran cmake build-essential
```

Next you will want to install Rust:

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
rustup set profile minimal
rustup toolchain add nightly-2021-11-02
```

Then clone and enter enter the `bqskitrs` directory:

```bash
git clone https://github.com/BQSKit/bqskitrs.git
cd bqskitrs
```

Then install the `bqskitrs` package:

```bash
pip install .
```

This will take a while. Once it is done, verify that the installation succeeded by running

```
python3 -c 'import bqskitrs'
```

It should not print anything out nor give any error.


### MacOS


Make sure the version of pip you have is at least 20.0.2.
You can check via `pip -V` and upgrade via `python3 -m pip install -U pip`.

First, install the dependencies. We use homebrew here, which is what we build the official package against.

```
brew install gcc eigen lapack
```

Once that is complete you should then install Rust like as follows:

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
rustup set profile minimal
rustup toolchain add nightly-2021-11-02
```

Then clone and enter enter the `bqskitrs` directory:

```bash
git clone https://github.com/BQSKit/bqskitrs.git
cd bqskitrs
```

Then build the wheel (package file) with [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
maturin build --cargo-extra-args="--no-default-features --features python,accelerate,ceres/static,mimalloc/local_dynamic_tls" --release --no-sdist
```

Finally install the wheel that you built:

```bash
pip install --no-index --find-links=target/wheels bqskitrs
```

Once it is done installing the wheel, verify that the installation succeeded by running

```
python3 -c 'import bqskitrs'
```

It should not print anything out nor give any error.

### Windows


Make sure the version of pip you have is at least 20.0.2.
You can check via `pip -V` and upgrade via `python3 -m pip install -U pip`.

Download and install rust via the installer found at https://rustup.rs/. Accept all of the defaults.

Close your shell and open a new one (this updates the enviroment). Then run

```shell
rustup toolchain add nightly-2021-11-02
```

Then install `cargo-vcpkg`, which will help install dependencies for us. You can install it via

```shell
cargo install cargo-vcpkg
```

`cargo` is installed with Rust so this should "just work".

Then clone and enter enter the `bqskitrs` directory:

```shell
git clone https://github.com/BQSKit/bqskitrs.git
cd bqskitrs
```

Then build the dependencies with vcpkg.

```shell
cargo vcpkg build
```

This will take quite a while.

Then build the wheel (package file) with [maturin](https://github.com/PyO3/maturin):

```shell
pip install maturin
maturin build --cargo-extra-args="--no-default-features --features python,static,openblas-src/system,mimalloc/local_dynamic_tls" --release --no-sdist
```

Finally install the wheel that you built:

```bash
pip install --no-index --find-links=target/wheels bqskitrs
```

Once it is done installing the wheel, verify that the installation succeeded by running

```
python3 -c 'import bqskitrs'
```

It should not print anything out nor give any error.

## Benchmarking

This crate also supports benchmarking changes. Once you have dependencies installed,
you should install `cargo-criterion` to run benchmarks:

```bash
cargo install cargo-criterion
```

Then you can run a command like the following to benchmark the instantiators:

```bash
cargo criterion --no-default-features --features openblas-src,blas-src/openblas,openblas-src/system,openblas-src/cblas,openblas-src/lapacke,squaremat/openblas-system,mimalloc
```
