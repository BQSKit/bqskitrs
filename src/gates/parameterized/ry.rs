use crate::gates::utils::{rot_y, rot_y_jac};
use crate::gates::Unitary;
use crate::gates::{Gradient, Size};

use squaremat::SquareMatrix;

/// Arbitrary Y rotation single qubit gate
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct RYGate();

impl RYGate {
    pub fn new() -> Self {
        RYGate {}
    }
}

impl Unitary for RYGate {
    fn num_params(&self) -> usize {
        1
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        rot_y(params[0])
    }
}

impl Gradient for RYGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[SquareMatrix]) -> Vec<SquareMatrix> {
        vec![rot_y_jac(params[0])]
    }
}

impl Size for RYGate {
    fn get_size(&self) -> usize {
        1
    }
}
