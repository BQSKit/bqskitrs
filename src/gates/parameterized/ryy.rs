use crate::gates::{Gradient, Size};
use crate::gates::{Optimize, Unitary};
use crate::{i, r};

use ndarray::ArrayViewMut2;
use num_complex::Complex64;
use squaremat::SquareMatrix;

/// A gate representing an arbitrary rotation around the YY axis
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct RYYGate();

impl RYYGate {
    pub fn new() -> Self {
        RYYGate {}
    }
}

impl Unitary for RYYGate {
    fn num_params(&self) -> usize {
        1
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        let cos = r!((params[0] / 2.).cos());
        let nsin = i!(-1.0) * (params[0] / 2.).sin();
        let psin = i!(1.0) * (params[0] / 2.).sin();
        let zero = r!(0.0);
        SquareMatrix::from_vec(
            vec![
                cos, zero, zero, psin, zero, cos, nsin, zero, zero, nsin, cos, zero, psin, zero,
                zero, cos,
            ],
            4,
        )
    }
}

impl Gradient for RYYGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[SquareMatrix]) -> Vec<SquareMatrix> {
        let dcos = -1. * r!(params[0] / 2.).sin() / 2.;
        let dnsin = i!(-1.0) * (params[0] / 2.).cos() / 2.;
        let dpsin = i!(1.0) * (params[0] / 2.).cos() / 2.;
        let zero = r!(0.0);
        vec![SquareMatrix::from_vec(
            vec![
                dcos, zero, zero, dpsin, zero, dcos, dnsin, zero, zero, dnsin, dcos, zero, dpsin,
                zero, zero, dcos,
            ],
            4,
        )]
    }
}

impl Size for RYYGate {
    fn get_size(&self) -> usize {
        2
    }
}

impl Optimize for RYYGate {
    fn optimize(&self, _env_matrix: ArrayViewMut2<Complex64>) -> Vec<f64> {
        unimplemented!()
    }
}
