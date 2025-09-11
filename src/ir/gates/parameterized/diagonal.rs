use crate::ir::gates::utils::imag_exp;
use crate::ir::gates::{Gradient, Size};
use crate::ir::gates::{Optimize, Unitary};

use ndarray::{Array2, Array3, ArrayViewMut2};
use ndarray_linalg::c64;



/// A gate representing a Diagonal Gate on n-qubits
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct DiagonalGate {
    size: usize,
    dim: usize,
    shape: (usize, usize),
    num_parameters: usize,
}

impl DiagonalGate {
    pub fn new(size: usize) -> Self {
        let base: u32 = 2;
        let dim = base.pow(size as u32) as usize;
        let num_params = (base.pow((size) as u32) - 1) as usize;
        DiagonalGate {
            size,
            dim,
            shape: (dim, dim),
            num_parameters: num_params,
        }
    }
}

impl Unitary for DiagonalGate {
    fn num_params(&self) -> usize {
        self.num_parameters
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<c64>]) -> Array2<c64> {
        let mut arr: Array2<c64> = Array2::zeros((self.dim, self.dim));
        let mut i: usize = 1;
        arr[[0, 0]] = c64::new(1 as f64, 0 as f64);
        for param in params {
            arr[[i, i]] = imag_exp(*param);
            i += 1;
        }
        arr
    }
}

impl Gradient for DiagonalGate {
    fn get_grad(&self, _params: &[f64], _const_gates: &[Array2<c64>]) -> Array3<c64> {
        let mut grad: Array3<c64> = Array3::zeros((_params.len(), self.dim, self.dim));

        for (i, param) in _params.iter().enumerate() {
            let ind = i + 1;
            grad[[i, ind, ind]] = c64::new(0.0, 1.0) * imag_exp(*param);
        }

        grad
    }

    fn get_utry_and_grad(
        &self,
        _params: &[f64],
        _const_gates: &[Array2<c64>],
    ) -> (Array2<c64>, Array3<c64>) {
        let utry = self.get_utry(_params, _const_gates);
        let grad = self.get_grad(_params, _const_gates);
        (utry, grad)
    }
}

impl Size for DiagonalGate {
    fn num_qudits(&self) -> usize {
        self.size
    }
}

impl Optimize for DiagonalGate {
    fn optimize(&self, env_matrix: ArrayViewMut2<c64>) -> Vec<f64> {
        let mut thetas: Vec<f64> = Vec::new();
        let mut i: usize = 1;
        let real_0 = env_matrix[[0, 0]].re;
        let imag_0 = env_matrix[[0, 0]].im;
        // We want the top left element to be 1, so the current top left
        // element is the global phase offset
        let global_phase = (imag_0 / real_0).atan();
        // Calculate each theta and subtract from the global phase
        while i < self.dim {
            let real = env_matrix[[i, i]].re;
            let imag = env_matrix[[i, i]].im;
            let phase = (imag / real).atan();
            let a = global_phase - phase;
            thetas.push(a);
            i = i + 1;
        }
        thetas
    }
}
