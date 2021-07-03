use crate::gates::utils::{rot_y, rot_y_jac};
use crate::gates::{Gradient, Size};
use crate::gates::{Optimize, Unitary};

use ndarray::{Array2, Array3, ArrayViewMut2, Axis};
use num_complex::Complex64;

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

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<Complex64>]) -> Array2<Complex64> {
        rot_y(params[0])
    }
}

impl Gradient for RYGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[Array2<Complex64>]) -> Array3<Complex64> {
        let mut out = Array3::zeros((1, 2, 2));
        let mut arr = out.index_axis_mut(Axis(0), 0);
        arr.assign(&rot_y_jac(params[0]));
        out
    }
}

impl Size for RYGate {
    fn get_size(&self) -> usize {
        1
    }
}

impl Optimize for RYGate {
    fn optimize(&self, env_matrix: ArrayViewMut2<Complex64>) -> Vec<f64> {
        let real = (env_matrix[[0, 0]] + env_matrix[[1, 1]]).re;
        let imag = (env_matrix[[1, 0]] - env_matrix[[0, 1]]).im;
        let mut theta = 2.0 * (real / (real.powi(2) + imag.powi(2)).sqrt()).acos();
        theta *= if imag < 0.0 { -1.0 } else { 1.0 };
        vec![theta; 1]
    }
}
