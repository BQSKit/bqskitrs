use crate::{gates::{Gradient, Unitary}, operation::Operation, permutation_matrix::calc_permutation_matrix, unitary_builder::UnitaryBuilder};

use squaremat::SquareMatrix;

/// A list of gates in a quantum circuit
#[derive(Clone)]
pub struct Circuit {
    pub size: usize,
    pub radixes: Vec<usize>,
    pub ops: Vec<Operation>,
    pub constant_gates: Vec<SquareMatrix>,
}

impl Circuit {
    pub fn new(
        size: usize,
        radixes: Vec<usize>,
        ops: Vec<Operation>,
        constant_gates: Vec<SquareMatrix>,
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
        self.ops.iter().fold(ret, |mut ret, op| {ret.extend_from_slice(&op.params); ret})
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

    fn get_utry(&self, params: &[f64], const_gates: &[SquareMatrix]) -> SquareMatrix {
        if !params.is_empty() {
            assert_eq!(params.len(), self.num_params());
            let mut param_idx = 0;
            let mut builder = UnitaryBuilder::new(self.size, self.radixes.clone());
            for op in &self.ops {
                let utry = op.get_utry(
                    &params[param_idx..param_idx + op.num_params()],
                    const_gates,
                );
                param_idx += op.num_params();
                builder.apply_right(&utry, &op.location, false);
            }
            builder.get_utry()
        } else {
            let mut builder = UnitaryBuilder::new(self.size, self.radixes.clone());
            for op in &self.ops {
                let utry = op.get_utry(
                    &[],
                    const_gates,
                );
                builder.apply_right(&utry, &op.location, false);
            }
            builder.get_utry()
        }
    }
}

impl Gradient for Circuit {
    fn get_grad(&self, params: &[f64], const_gates: &[SquareMatrix]) -> Vec<SquareMatrix> {
        self.get_utry_and_grad(params, const_gates).1
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        const_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        let mut matrices = vec![];
        let mut grads = vec![];
        let mut locations = vec![];
        if params.is_empty() {
            for op in &self.ops {
                let (utry, grad) = op.get_utry_and_grad(
                    &[],
                    const_gates,
                );
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
                param_idx += op.num_params();
                matrices.push(utry);
                grads.push(grad);
                locations.push(&op.location);
            }
        }

        let mut left = UnitaryBuilder::new(self.size, self.radixes.clone());
        let mut right = UnitaryBuilder::new(self.size, self.radixes.clone());
        let mut full_grads = vec![];
        for (m, location) in matrices.iter().zip(locations.iter()) {
            right.apply_right(m, location, false);
        }

        for ((m, location), d_m) in matrices.iter().zip(locations.iter()).zip(grads) {
            let perm = calc_permutation_matrix(self.size, (*location).clone());
            let perm_t = perm.T();
            let id = SquareMatrix::eye(2usize.pow((self.size - location.len()) as u32));

            right.apply_left(m, location, true);
            let right_utry = right.get_utry();
            let left_utry = left.get_utry();
            for grad in d_m {
                let mut full_grad = grad.kron(&id);
                full_grad = perm.matmul(&full_grad);
                full_grad = full_grad.matmul(&perm_t);
                let right_grad = right_utry.matmul(&full_grad);
                full_grads.push(right_grad.matmul(&left_utry));
            }
            left.apply_right(&m, location, false);
        }
        (left.get_utry(), full_grads)
    }
}
