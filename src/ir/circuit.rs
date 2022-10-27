use crate::qis::unitary::UnitaryBuilder;
use super::gates::{Gate, Gradient, Unitary};
use super::Operation;
use itertools::Itertools;

use itertools::izip;
use ndarray::{Array2, Array3, ArrayD, ArrayView2, Ix2};
use ndarray_linalg::c64;
use crate::squaremat::*;
use crate::permutation_matrix::calc_permutation_matrix;

type Cycle = usize;

#[derive(Clone)]
pub struct Circuit {
    pub size: usize,
    pub radixes: Vec<usize>,
    pub ops: Vec<Operation>,
    pub constant_gates: Vec<Array2<c64>>,
    pub cycle_boundaries: Vec<(usize, usize)>,
    pub num_params: usize,
    pub sendable: bool,
    pub dim: usize,
}

impl Circuit {
    pub fn new(
        size: usize,
        radixes: Vec<usize>,
        ops_with_cycles: Vec<(Cycle, Operation)>,
        constant_gates: Vec<Array2<c64>>,
    ) -> Self {
        let mut sendable = true;
        let mut num_params = 0;
        let mut cycle_boundaries = Vec::new();
        let mut current_cycle = 0usize;
        for (cycle, op) in &ops_with_cycles {
            num_params += op.gate.num_params();
            if let Gate::Dynamic(_) = op.gate {
                sendable = false
            }
            if *cycle != current_cycle {
                cycle_boundaries.push((current_cycle, *cycle));
                current_cycle += 1;
            }
        }
        let dim: usize = radixes.iter().product();
        Circuit {
            size,
            radixes,
            ops: ops_with_cycles.iter().map(|(_, op)| op.clone()).collect(),
            constant_gates,
            cycle_boundaries,
            num_params,
            sendable,
            dim,
        }
    }

    pub fn is_sendable(&self) -> bool {
        self.sendable
    }

    pub fn get_params(&self) -> Vec<f64> {
        let ret = Vec::with_capacity(self.num_params());
        self.ops.iter().fold(ret, |mut ret, op| {
            ret.extend_from_slice(&op.params);
            ret
        })
    }

    pub fn set_params(&mut self, params: &[f64]) {
        if params.len() != self.num_params() {
            panic!(
                "Incorrect number of parameters in set_params, expected {} got {}",
                self.num_params(),
                params.len()
            );
        }
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
        self.num_params
    }

    fn get_utry(&self, params: &[f64], const_gates: &[Array2<c64>]) -> Array2<c64> {
        if !params.is_empty() {
            let mut param_idx = 0;
            let mut builder = UnitaryBuilder::new(self.size, self.radixes.clone());
            for op in &self.ops {
                let utry = op
                    .get_utry(&params[param_idx..param_idx + op.num_params()], const_gates);
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
    fn get_utry_and_grad(
        &self,
        params: &[f64],
        const_gates: &[Array2<c64>],
    ) -> (Array2<c64>, Array3<c64>) {
        if params.len() != self.num_params() {
            panic!(
                "Incorrect number of params passed to circuit, expected {}, got {}",
                self.num_params(),
                params.len()
            );
        }
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
        let mut out_grad = Array3::zeros((
            num_grads,
            self.dim,
            self.dim,
        ));
        let mut out_iter = out_grad.outer_iter_mut();

        for (m, location) in matrices.iter().zip(locations.iter()) {
            right.apply_right(m.view(), location, false);
        }

        for (m, location, d_m) in izip!(matrices, locations, grads) {
            // 1. Permute and reshape right tensor for location
            let right_perm: Vec<usize> = location.iter().map(|x| x + right.num_qudits).collect();
            let left_perm = (0..right.num_idxs).filter(|x| !right_perm.contains(&x));
            let mut perm = vec![];
            perm.extend(left_perm);
            perm.extend(right_perm);
            right.permute_idxs(perm);

            let owned_tensor = right.tensor.take().unwrap();
            let shape = right.get_current_shape();
            let right_dim: usize = shape[shape.len()-location.len()..].iter().product();
            let reshaped = owned_tensor
                .to_shape((right.dim * right.dim / right_dim, right_dim))
                .expect("Cannot reshape tensor to matrix");
            
            // Apply inverse gate to the left of the right tensor
            let prod = reshaped.dot(&m.conj().t());
            let reshape_back = prod
                .into_shape(shape.clone())
                .expect("Failed to reshape matrix product back");
            right.tensor = Some(reshape_back.to_owned());

            // Store left unitary and then apply gate to left on right
            let left_utry = left.get_utry();
            left.apply_right(m.view(), location, false);

            // Compute un-permute order
            let revert: Vec<usize> = right.pi.clone().iter()
                .enumerate()
                .sorted_by(|(_idx_a, a), (_idx_b, b)| a.cmp(b))
                .map(|(idx, _a)| idx)
                .collect();

            // Calculate all gradients
            for grad in d_m.outer_iter() {
                let right_grad = reshaped.dot(&grad.view())
                    .into_shape(shape.clone())
                    .expect("Failed to reshape matrix product back")
                    .permuted_axes(revert.clone());
                let reshaped_grad = right_grad.to_shape((right.dim, right.dim))
                    .unwrap()
                    .into_dimensionality::<Ix2>()
                    .unwrap();
                let full_grad = reshaped_grad.dot(&left_utry.view());
                out_iter.next().unwrap().assign(&full_grad);
            }
        }

        (left.get_utry(), out_grad)
    }

    fn get_grad(&self, params: &[f64], const_gates: &[Array2<c64>]) -> Array3<c64> {
        if params.len() != self.num_params() {
            panic!(
                "Incorrect number of params passed to circuit, expected {}, got {}",
                self.num_params(),
                params.len()
            );
        }
        self.get_utry_and_grad(params, const_gates).1
    }
}