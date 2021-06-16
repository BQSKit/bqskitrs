use crate::gates::utils::{rot_z, rot_z_jac};
use crate::gates::Unitary;
use crate::gates::{Gradient, Size};

use squaremat::SquareMatrix;

/// Arbitrary Y rotation single qubit gate
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct RZGate();

impl RZGate {
    pub fn new() -> Self {
        RZGate {}
    }
}

impl Unitary for RZGate {
    fn num_params(&self) -> usize {
        1
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        rot_z(params[0])
    }
}

impl Gradient for RZGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[SquareMatrix]) -> Vec<SquareMatrix> {
        vec![rot_z_jac(params[0])]
    }
}

impl Size for RZGate {
    fn get_size(&self) -> usize {
        1
    }
}
