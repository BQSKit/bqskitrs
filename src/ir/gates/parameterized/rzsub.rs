use std::f64::consts::PI;

use crate::i;
use crate::ir::gates::utils::{rot_z, rot_z_jac};
use crate::ir::gates::{Gradient, Size};
use crate::ir::gates::{Optimize, Unitary};

use ndarray::{Array2, Array3, ArrayViewMut2};
use ndarray_linalg::c64;

/// Arbitrary Y rotation single qubit gate
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct RZSubGate{
    radix: usize,
    level1: usize,
    level2: usize,
}

impl RZSubGate {
    pub fn new(radix: usize, level1: usize, level2: usize) -> Self {
        RZSubGate {
            radix: radix,
            level1: level1,
            level2: level2,
        }
    }
}

impl Unitary for RZSubGate {
    fn num_params(&self) -> usize {
        1
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<c64>]) -> Array2<c64> {
        let pexp = i!(0.5 * params[0]).exp();
        let nexp = i!(-0.5 * params[0]).exp();

        let mut unitary = Array2::eye(self.radix);
        unitary[[self.level1, self.level1]] = nexp;
        unitary[[self.level2, self.level2]] = pexp;
        unitary
    }
}

impl Gradient for RZSubGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[Array2<c64>]) -> Array3<c64> {
        let dpexp = i!(0.5) * i!(0.5 * params[0]).exp();
        let dnexp = i!(-0.5) * i!(-0.5 * params[0]).exp();

        let mut grad = Array3::zeros((1, self.radix, self.radix));
        grad[[0, self.level1, self.level1]] = dnexp;
        grad[[0, self.level2, self.level2]] = dpexp;
        grad
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        _const_gates: &[Array2<c64>],
    ) -> (Array2<c64>, Array3<c64>) {
        let pexp = i!(0.5 * params[0]).exp();
        let nexp = i!(-0.5 * params[0]).exp();
        let dpexp = i!(0.5) * pexp;
        let dnexp = i!(-0.5) * nexp;

        let mut unitary = Array2::eye(self.radix);
        unitary[[self.level1, self.level1]] = nexp;
        unitary[[self.level2, self.level2]] = pexp;

        let mut grad = Array3::zeros((1, self.radix, self.radix));
        grad[[0, self.level1, self.level1]] = dnexp;
        grad[[0, self.level2, self.level2]] = dpexp;

        (unitary, grad)
    }
}

impl Size for RZSubGate {
    fn num_qudits(&self) -> usize {
        1
    }
}

impl Optimize for RZSubGate {
    fn optimize(&self, env_matrix: ArrayViewMut2<c64>) -> Vec<f64> {
        unimplemented!()
    }
}
