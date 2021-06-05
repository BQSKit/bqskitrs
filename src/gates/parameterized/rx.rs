use crate::gates::utils::{rot_x, rot_x_jac};
use crate::gates::Unitary;
use crate::gates::{Gradient, Size};

use squaremat::SquareMatrix;

/// Arbitrary X rotation single qubit gate
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RXGate();

impl RXGate {
    pub fn new() -> Self {
        RXGate {}
    }
}

impl Unitary for RXGate {
    fn num_params(&self) -> usize {
        1
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        rot_x(params[0])
    }
}

impl Gradient for RXGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[SquareMatrix]) -> Vec<SquareMatrix> {
        vec![rot_x_jac(params[0])]
    }
}

impl Size for RXGate {
    fn get_size(&self) -> usize {
        1
    }
}
