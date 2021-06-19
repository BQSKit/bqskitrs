use std::f64::consts::PI;

use crate::gates::utils::{rot_z, rot_z_jac};
use crate::gates::{Gradient, Size};
use crate::gates::{Optimize, Unitary};
use crate::i;

use ndarray::ArrayViewMut2;
use num_complex::Complex64;
use squaremat::SquareMatrix;

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

    fn get_utry(&self, params: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        let phase = (i!(1.0) * params[0] / 2.0).exp();
        rot_z(params[0]) * phase
    }
}

impl Gradient for U1Gate {
    fn get_grad(&self, params: &[f64], _const_gates: &[SquareMatrix]) -> Vec<SquareMatrix> {
        let phase = (i!(1.0) * params[0] / 2.0).exp();
        let dphase = i!(1.0) / 2.0 * phase;
        vec![rot_z(params[0]) * dphase + rot_z_jac(params[0]) * phase]
    }
}

impl Size for U1Gate {
    fn get_size(&self) -> usize {
        1
    }
}

impl Optimize for U1Gate {
    fn optimize(&self, env_matrix: ArrayViewMut2<Complex64>) -> Vec<f64> {
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
