pub mod qis;
pub mod ir;
pub mod utils;
pub mod squaremat;
pub mod python;
pub mod permutation_matrix;

#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(any(feature = "default", feature = "openblas"))]
#[link(name = "openblas")]
extern "C" {}

#[macro_export]
macro_rules! c {
    ($re:expr, $im:expr) => {
        c64::new($re, $im)
    };
}

#[macro_export]
macro_rules! r {
    ($re:expr) => {
        c64::new($re, 0.0)
    };
}

#[macro_export]
macro_rules! i {
    ($im:expr) => {
        c64::new(0.0, $im)
    };
}
