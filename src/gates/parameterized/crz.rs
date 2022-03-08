use crate::gates::{Gradient, Size};
use crate::gates::{Optimize, Unitary};
use crate::{i, r};

use ndarray::{Array2, Array3, ArrayViewMut2};
use num_complex::Complex64;

/// A gate representing an arbitrary rotation around the ZZ axis
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct CRZGate();

impl CRZGate {
    pub fn new() -> Self {
        CRZGate {}
    }
}

impl Unitary for CRZGate {
    fn num_params(&self) -> usize {
        1
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<Complex64>]) -> Array2<Complex64> {
        let pos = (i!(1.) * params[0] / 2.).exp();
        let neg = (i!(-1.) * params[0] / 2.).exp();
        let zero = r!(0.0);
        let one = r!(1.0);
        Array2::from_shape_vec(
            (4, 4),
            vec![
                one, zero, zero, zero, zero, one, zero, zero, zero, zero, neg, zero, zero, zero,
                zero, pos,
            ],
        )
        .unwrap()
    }
}

impl Gradient for CRZGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[Array2<Complex64>]) -> Array3<Complex64> {
        let zero = r!(0.0);
        let dpos = i!(1. / 2.) * (i!(1.) * params[0] / 2.).exp();
        let dneg = i!(-1. / 2.) * (i!(-1.) * params[0] / 2.).exp();
        Array3::from_shape_vec(
            (1, 4, 4),
            vec![
                zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, dneg, zero, zero, zero,
                zero, dpos,
            ],
        )
        .unwrap()
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        _const_gates: &[Array2<Complex64>],
    ) -> (Array2<Complex64>, Array3<Complex64>) {
        let pos = (i!(1.) * params[0] / 2.).exp();
        let neg = (i!(-1.) * params[0] / 2.).exp();
        let zero = r!(0.0);
        let dpos = i!(1. / 2.) * (i!(1.) * params[0] / 2.).exp();
        let dneg = i!(-1. / 2.) * (i!(-1.) * params[0] / 2.).exp();
        let one = r!(1.0);

        (
            Array2::from_shape_vec(
                (4, 4),
                vec![
                    one, zero, zero, zero, zero, one, zero, zero, zero, zero, neg, zero, zero,
                    zero, zero, pos,
                ],
            )
            .unwrap(),
            Array3::from_shape_vec(
                (1, 4, 4),
                vec![
                    zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, dneg, zero, zero,
                    zero, zero, dpos,
                ],
            )
            .unwrap(),
        )
    }
}

impl Size for CRZGate {
    fn num_qudits(&self) -> usize {
        2
    }
}

impl Optimize for CRZGate {
    fn optimize(&self, _env_matrix: ArrayViewMut2<Complex64>) -> Vec<f64> {
        unimplemented!()
    }
}
