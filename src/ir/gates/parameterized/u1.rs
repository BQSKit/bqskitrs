use std::f64::consts::PI;

use crate::ir::gates::utils::{rot_z, rot_z_jac};
use crate::ir::gates::{Gradient, Size};
use crate::ir::gates::{Optimize, Unitary};
use crate::i;

use ndarray::{Array2, Array3, ArrayViewMut2};
use ndarray_linalg::c64;

/// IBM's U1 single qubit gate
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct U1Gate();

impl U1Gate {
    pub fn new() -> Self {
        U1Gate {}
    }
}

impl Unitary for U1Gate {
    fn num_params(&self) -> usize {
        1
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<c64>]) -> Array2<c64> {
        let phase = (i!(1.0) * params[0] / 2.0).exp();
        rot_z(params[0], Some(phase))
    }
}

impl Gradient for U1Gate {
    fn get_grad(&self, params: &[f64], _const_gates: &[Array2<c64>]) -> Array3<c64> {
        let phase = (i!(1.0) * params[0] / 2.0).exp();
        let dphase = i!(1.0) / 2.0 * phase;
        rot_z(params[0], Some(dphase)) + rot_z_jac(params[0], Some(phase))
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        _const_gates: &[Array2<c64>],
    ) -> (Array2<c64>, Array3<c64>) {
        let phase = (i!(1.0) * params[0] / 2.0).exp();
        let dphase = i!(1.0) / 2.0 * phase;
        (
            rot_z(params[0], Some(phase)),
            rot_z(params[0], Some(dphase)) + rot_z_jac(params[0], Some(phase)),
        )
    }
}

impl Size for U1Gate {
    fn num_qudits(&self) -> usize {
        1
    }
}

impl Optimize for U1Gate {
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
