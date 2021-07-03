use crate::circuit::Circuit;
use enum_dispatch::enum_dispatch;

mod qfactor;

use ndarray::Array2;
use num_complex::Complex64;
pub use qfactor::QFactorInstantiator;

#[enum_dispatch]
pub trait Instantiate {
    fn instantiate(&self, circuit: Circuit, target: Array2<Complex64>, x0: &[f64]) -> Vec<f64>;
}

#[enum_dispatch(Instantiate)]
pub enum Instantiator {
    QFactor(QFactorInstantiator),
}
