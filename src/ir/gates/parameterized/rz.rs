use std::f64::consts::PI;

use crate::ir::gates::utils::{rot_z, rot_z_jac};
use crate::ir::gates::{Gradient, Size};
use crate::ir::gates::{Optimize, Unitary};

use ndarray::{Array2, Array3, ArrayViewMut2};
use ndarray_linalg::c64;

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

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<c64>]) -> Array2<c64> {
        rot_z(params[0], None)
    }
}

impl Gradient for RZGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[Array2<c64>]) -> Array3<c64> {
        rot_z_jac(params[0], None)
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        _const_gates: &[Array2<c64>],
    ) -> (Array2<c64>, Array3<c64>) {
        (rot_z(params[0], None), rot_z_jac(params[0], None))
    }
}

impl Size for RZGate {
    fn num_qudits(&self) -> usize {
        1
    }
}

impl Optimize for RZGate {
    fn optimize(&self, env_matrix: ArrayViewMut2<c64>) -> Vec<f64> {
        let real = env_matrix[[1, 1]].re;
        let imag = env_matrix[[1, 1]].im;
        let mut theta = (imag / real).atan();
        if real < 0.0 && imag > 0.0 {
            theta += PI;
        } else if real < 0.0 && imag < 0.0 {
            theta -= PI;
        }
        theta = -theta;
        vec![theta; 1]
    }
}
