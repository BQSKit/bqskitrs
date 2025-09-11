use crate::ir::gates::utils::{rot_y, get_indices, svd};
use crate::ir::gates::{Gradient, Size};
use crate::ir::gates::{Optimize, Unitary};
use crate::{i, r};

use ndarray::{Array2, Array3, ArrayViewMut2};
use ndarray_linalg::c64;
use crate::squaremat::Matmul;

/// A gate representing a multiplexed Y rotation with a target qudit
/// and n -1 select qudits
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct MPRYGate {
    size: usize,
    dim: usize,
    target_qudit: usize,
    shape: (usize, usize),
    num_parameters: usize,
}

impl MPRYGate {
    pub fn new(size: usize, target_qudit: usize) -> Self {
        let base: u32 = 2;
        let dim = base.pow(size as u32) as usize;
        let num_params = base.pow((size - 1) as u32) as usize;
        MPRYGate {
            size,
            dim,
            target_qudit,
            shape: (dim, dim),
            num_parameters: num_params,
        }
    }
}

impl Unitary for MPRYGate {
    fn num_params(&self) -> usize {
        self.num_parameters
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<c64>]) -> Array2<c64> {
        // Return the unitary matrix for the MPRY gate
        // If the target qudit is the bottom most qubit, then the matrix
        // is block diagonal of RY rotations
        // Otherwise, use the get_indices to permute the matrix appropriately
        let mut arr: Array2<c64> = Array2::zeros((self.dim, self.dim));
        let mut i: usize = 0;
        for param in params {
            let block = rot_y(*param);
            let (x1, x2) = get_indices(i, self.target_qudit, self.size);
            arr[[x1, x1]] = block[[0, 0]];
            arr[[x2, x2]] = block[[1, 1]];
            arr[[x2, x1]] = block[[1, 0]];
            arr[[x1, x2]] = block[[0, 1]];
            i += 1;
        }
        let (u, vt) = svd(arr.view_mut());
        u.matmul(vt.view())
    }
}

impl Gradient for MPRYGate {
    fn get_grad(&self, _params: &[f64], _const_gates: &[Array2<c64>]) -> Array3<c64> {
        // Each theta is independent, so the gradient is simply
        // the derivative of the rotation matrix with respect to theta
        let mut grad: Array3<c64> = Array3::zeros((_params.len(), self.dim, self.dim));

        for (i, &param) in _params.iter().enumerate() {
            let dcos = -((param / 2.0).sin()) / 2.0;
            let dsin = -c64::new(0.0, 1.0) * ((param / 2.0).cos()) / 2.0;

            let (x1, x2) = get_indices(i, self.target_qudit, self.size);

            grad[[i, x1, x1]] = c64::new(dcos, 0.0);
            grad[[i, x2, x2]] = c64::new(dcos, 0.0);
            grad[[i, x2, x1]] = dsin;
            grad[[i, x1, x2]] = -dsin;
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

impl Size for MPRYGate {
    fn num_qudits(&self) -> usize {
        self.size
    }
}

impl Optimize for MPRYGate {
    fn optimize(&self, env_matrix: ArrayViewMut2<c64>) -> Vec<f64> {
        let mut thetas: Vec<f64> = Vec::new();
        let mut i: usize = 0;
        // Each variable is independent, so you can optimize each Ry separately
        while i < self.dim {
            let (x1, x2) = get_indices(i, self.target_qudit, self.size);
            let a = (env_matrix[[x1, x1]] + env_matrix[[x2, x2]]).re;
            let b = (env_matrix[[x2, x1]] - env_matrix[[x1, x2]]).re;
            let mut theta = 2.0 * (a / (a.powi(2) + b.powi(2)).sqrt()).acos();
            theta *= if b > 0.0 { -1.0 } else { 1.0 };
            thetas.push(theta);
            i = i + 2;
        }
        thetas
    }
}
