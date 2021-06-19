use squaremat::SquareMatrix;

use crate::circuit::Circuit;
use enum_dispatch::enum_dispatch;

mod qfactor;

pub use qfactor::QFactorInstantiator;

#[enum_dispatch]
pub trait Instantiate {
    fn instantiate(&self, circuit: Circuit, target: SquareMatrix, x0: &[f64]) -> Vec<f64>;
}

#[enum_dispatch(Instantiate)]
pub enum Instantiator {
    QFactor(QFactorInstantiator),
}
