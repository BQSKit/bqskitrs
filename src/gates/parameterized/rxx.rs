use crate::gates::{Gradient, Size};
use crate::gates::{Optimize, Unitary};
use crate::{i, r};

use ndarray::ArrayViewMut2;
use num_complex::Complex64;
use squaremat::SquareMatrix;

/// A gate representing an arbitrary rotation around the XX axis
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct RXXGate();

impl RXXGate {
    pub fn new() -> Self {
        RXXGate {}
    }
}

impl Unitary for RXXGate {
    fn num_params(&self) -> usize {
        1
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        let cos = r!((params[0] / 2.).cos());
        let sin = i!(-1.0) * (params[0] / 2.).sin();
        let zero = r!(0.0);
        SquareMatrix::from_vec(
            vec![
                cos, zero, zero, sin, zero, cos, sin, zero, zero, sin, cos, zero, sin, zero, zero,
                cos,
            ],
            4,
        )
    }
}

impl Gradient for RXXGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[SquareMatrix]) -> Vec<SquareMatrix> {
        let dcos = -1. * r!(params[0] / 2.).sin() / 2.;
        let dsin = i!(-1.0) * (params[0] / 2.).cos() / 2.;
        let zero = r!(0.0);
        vec![SquareMatrix::from_vec(
            vec![
                dcos, zero, zero, dsin, zero, dcos, dsin, zero, zero, dsin, dcos, zero, dsin, zero,
                zero, dcos,
            ],
            4,
        )]
    }
}

impl Size for RXXGate {
    fn get_size(&self) -> usize {
        2
    }
}

impl Optimize for RXXGate {
    fn optimize(&self, env_matrix: ArrayViewMut2<Complex64>) -> Vec<f64> {
        let re = env_matrix.diag().sum().re;
        let im =
            (env_matrix[[0, 3]] + env_matrix[[1, 2]] + env_matrix[[2, 1]] + env_matrix[[3, 0]]).im;
        let mut theta = (re / (re.powi(2) + im.powi(2)).sqrt()).acos();
        if im < 0. {
            theta *= -2.;
        } else {
            theta *= 2.;
        }
        vec![theta]
    }
}
