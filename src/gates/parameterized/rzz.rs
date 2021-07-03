use crate::gates::{Gradient, Size};
use crate::gates::{Optimize, Unitary};
use crate::{i, r};

use ndarray::{Array2, Array3, ArrayViewMut2};
use num_complex::Complex64;

/// A gate representing an arbitrary rotation around the ZZ axis
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct RZZGate();

impl RZZGate {
    pub fn new() -> Self {
        RZZGate {}
    }
}

impl Unitary for RZZGate {
    fn num_params(&self) -> usize {
        1
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<Complex64>]) -> Array2<Complex64> {
        let pos = (i!(1.) * params[0] / 2.).exp();
        let neg = (i!(-1.) * params[0] / 2.).exp();
        let zero = r!(0.0);
        Array2::from_shape_vec(
            (4, 4),
            vec![
                neg, zero, zero, zero, zero, pos, zero, zero, zero, zero, pos, zero, zero, zero,
                zero, neg,
            ],
        )
        .unwrap()
    }
}

impl Gradient for RZZGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[Array2<Complex64>]) -> Array3<Complex64> {
        let dpos = i!(1. / 2.) * (i!(1.) * params[0] / 2.).exp();
        let dneg = i!(-1. / 2.) * (i!(-1.) * params[0] / 2.).exp();
        let zero = r!(0.0);
        Array3::from_shape_vec(
            (1, 4, 4),
            vec![
                dneg, zero, zero, zero, zero, dpos, zero, zero, zero, zero, dpos, zero, zero, zero,
                zero, dneg,
            ],
        )
        .unwrap()
    }
}

impl Size for RZZGate {
    fn get_size(&self) -> usize {
        2
    }
}

impl Optimize for RZZGate {
    fn optimize(&self, _env_matrix: ArrayViewMut2<Complex64>) -> Vec<f64> {
        unimplemented!()
    }
}
