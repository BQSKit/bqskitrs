use crate::ir::circuit::Circuit;
use enum_dispatch::enum_dispatch;

mod qfactor;
pub mod minimizers;

use ndarray::Array2;
use ndarray_linalg::c64;
pub use qfactor::QFactorInstantiator;

#[enum_dispatch]
pub trait Instantiate {
    fn instantiate(&self, circuit: &mut Circuit, target: Array2<c64>, x0: &[f64])
        -> Vec<f64>;
}

#[enum_dispatch(Instantiate)]
pub enum Instantiator {
    QFactor(QFactorInstantiator),
}
