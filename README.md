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

It's best to build and install the manylinux version with the provided
docker container:

```bash
git clone https://github.com/BQSKit/bqskitrs.git --recursive
cd bqskitrs
docker run -e OPENBLAS_ARGS="DYNAMIC_ARCH=1" --rm -v $(pwd):/io edyounis/bqskitrs-manylinux:1.1 build  --release --features=openblas --compatibility=manylinux2014
pip install --no-index --find-links=target/wheels bqskitrs
```

### MacOS

Make sure the version of pip you have is at least 20.0.2.
You can check via `pip -V` and upgrade via `python3 -m pip install -U pip`.

First, install the dependencies. We use homebrew here, which is what we build the official package against.

```
brew install gcc ceres-solver eigen lapack
```

Once that is complete you should then install Rust like as follows:

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
```

Then clone and enter enter the `bqskitrs` directory:

```bash
git clone https://github.com/BQSKit/bqskitrs.git --recursive
cd bqskitrs
```

Then build the wheel (package file) with [maturin](https://github.com/PyO3/maturin):

```bash
pip install -U setuptools wheel maturin
maturin build --features="accelerate" --release
```

If you encounter issues on an intel x86 Mac computer with a message like "rust failed to run custom build command for cxx', you may need to run the following build command instead:

```bash
MACOSX_DEPLOYMENT_TARGET=11.0 maturin build --features="accelerate"
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

Close your shell and open a new one (this updates the enviroment).

Then install `cargo-vcpkg`, which will help install dependencies for us. You can install it via

```shell
cargo install cargo-vcpkg
```

`cargo` is installed with Rust so this should "just work".

Then clone and enter enter the `bqskitrs` directory:

```shell
git clone https://github.com/BQSKit/bqskitrs.git --recursive
cd bqskitrs
```

Then build the dependencies with vcpkg.

```shell
cargo vcpkg build
```

This will take quite a while.

Then build the wheel (package file) with [maturin](https://github.com/PyO3/maturin):

```shell
python -m pip install maturin
python -m maturin build --interpreter $(which python) --features="openblas,openblas-src/system" --release
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
