mod circuit;
mod gates;
mod instantiators;
mod minimizers;
mod operation;
mod permutation_matrix;
mod python;
mod unitary_builder;
mod utils;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
#[cfg(any(feature = "static", feature = "default"))]
extern crate openblas_src;

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
