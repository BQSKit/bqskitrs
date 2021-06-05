mod circuit;
mod gates;
#[cfg(feature = "python")]
mod python;
#[cfg(any(feature = "ceres", feature = "bfgs"))]
mod solvers;
mod unitary_builder;
//mod tensor_network;
mod instantiator;
mod utils;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(any(feature = "static", feature = "default"))]
extern crate openblas_src;

use circuit::Circuit;
use unitary_builder::UnitaryBuilder;

#[macro_export]
macro_rules! c {
    ($re:expr, $im:expr) => {
        Complex64::new($re, $im)
    };
}

#[macro_export]
macro_rules! r {
    ($re:expr) => {
        Complex64::new($re, 0.0)
    };
}

#[macro_export]
macro_rules! i {
    ($im:expr) => {
        Complex64::new(0.0, $im)
    };
}
