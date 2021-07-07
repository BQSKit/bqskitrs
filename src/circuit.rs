use crate::{
    gates::{Gradient, Unitary},
    operation::Operation,
    permutation_matrix::calc_permutation_matrix,
    unitary_builder::UnitaryBuilder,
};

use itertools::izip;

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use squaremat::*;

/// A list of gates in a quantum circuit
#[derive(Clone)]
pub struct Circuit {
    pub size: usize,
    pub radixes: Vec<usize>,
    pub ops: Vec<Operation>,
    pub constant_gates: Vec<Array2<Complex64>>,
}

impl Circuit {
    pub fn new(
        size: usize,
        radixes: Vec<usize>,
        ops: Vec<Operation>,
        constant_gates: Vec<Array2<Complex64>>,
    ) -> Self {
        Circuit {
            size,
            radixes,
            ops,
            constant_gates,
        }
    }

    pub fn get_params(&self) -> Vec<f64> {
        let ret = Vec::with_capacity(self.num_params());
        self.ops.iter().fold(ret, |mut ret, op| {
            ret.extend_from_slice(&op.params);
            ret
        })
    }

    pub fn set_params(&mut self, params: &[f64]) {
        let mut param_idx = 0;
        for op in self.ops.iter_mut() {
            let parameters = &params[param_idx..param_idx + op.num_params()];
            op.params.copy_from_slice(parameters);
            param_idx += op.num_params();
        }
    }
}

impl Unitary for Circuit {
    fn num_params(&self) -> usize {
        self.ops.iter().map(|i| i.gate.num_params()).sum()
    }

    fn get_utry(&self, params: &[f64], const_gates: &[Array2<Complex64>]) -> Array2<Complex64> {
        if !params.is_empty() {
            assert_eq!(params.len(), self.num_params());
            let mut param_idx = 0;
            let mut builder = UnitaryBuilder::new(self.size, self.radixes.clone());
            for op in &self.ops {
                let utry =
                    op.get_utry(&params[param_idx..param_idx + op.num_params()], const_gates);
                param_idx += op.num_params();
                builder.apply_right(utry.view(), &op.location, false);
            }
            builder.get_utry()
        } else {
            let mut builder = UnitaryBuilder::new(self.size, self.radixes.clone());
            for op in &self.ops {
                let utry = op.get_utry(&[], const_gates);
                builder.apply_right(utry.view(), &op.location, false);
            }
            builder.get_utry()
        }
    }
}

impl Gradient for Circuit {
    fn get_grad(&self, params: &[f64], const_gates: &[Array2<Complex64>]) -> Array3<Complex64> {
        self.get_utry_and_grad(params, const_gates).1
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        const_gates: &[Array2<Complex64>],
    ) -> (Array2<Complex64>, Array3<Complex64>) {
        let mut matrices = vec![];
        let mut grads = vec![];
        let mut locations = vec![];
        let mut num_grads = 0usize;
        if params.is_empty() {
            for op in &self.ops {
                let (utry, grad) = op.get_utry_and_grad(&[], const_gates);
                num_grads += grad.shape()[0];
                matrices.push(utry);
                grads.push(grad);
                locations.push(&op.location);
            }
        } else {
            let mut param_idx = 0;
            for op in &self.ops {
                let (utry, grad) = op.get_utry_and_grad(
                    &params[param_idx..param_idx + op.num_params()],
                    const_gates,
                );
                num_grads += grad.shape()[0];
                param_idx += op.num_params();
                matrices.push(utry);
                grads.push(grad);
                locations.push(&op.location);
            }
        }

        let mut left = UnitaryBuilder::new(self.size, self.radixes.clone());
        let mut right = UnitaryBuilder::new(self.size, self.radixes.clone());
        let mut full_grads = Vec::with_capacity(num_grads);
        let mut out_grad = Array3::zeros((
            num_grads,
            2usize.pow(self.size as u32),
            2usize.pow(self.size as u32),
        ));
        for (m, location) in matrices.iter().zip(locations.iter()) {
            right.apply_right(m.view(), location, false);
        }

        for (m, location, d_m) in izip!(matrices, locations, grads) {
            let perm = calc_permutation_matrix(self.size, (*location).clone());
            let perm_t = perm.clone().reversed_axes();
            let id = Array2::eye(2usize.pow((self.size - location.len()) as u32));

            right.apply_left(m.view(), location, true);
            let right_utry = right.get_utry();
            let left_utry = left.get_utry();
            for grad in d_m.outer_iter() {
                let mut full_grad = grad.kron(&id);
                full_grad = perm.dot(&full_grad);
                full_grad = full_grad.dot(&perm_t);
                let right_grad = right_utry.dot(&full_grad);
                full_grads.push(right_grad.dot(&left_utry));
            }
            left.apply_right(m.view(), location, false);
        }

        for (mut arr, grad) in out_grad.outer_iter_mut().zip(full_grads) {
            arr.assign(&grad);
        }

        (left.get_utry(), out_grad)
    }
}
