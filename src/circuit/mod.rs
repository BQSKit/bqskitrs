use crate::{
    gates::{Gate, Gradient, Unitary},
    permutation_matrix::calc_permutation_matrix,
    unitary_builder::UnitaryBuilder,
};

use itertools::izip;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use squaremat::*;

pub mod operation;

use operation::Operation;

use crate::permutation_matrix::permute_unitary;

#[derive(Clone)]
pub enum SimulationBackend {
    Tensor,
    Matrix,
}

type Cycle = usize;

fn kron_ops(
    circuit_size: usize,
    ops: &[Operation],
    params: &[f64],
    const_gates: &[Array2<Complex64>],
) -> (usize, Array2<Complex64>) {
    let first_op = &ops[0];
    let mut result = first_op.get_utry(&params[..first_op.num_params()], const_gates);
    result = permute_unitary(result.view(), circuit_size, first_op.location.clone());
    let mut index = first_op.num_params();
    for op in &ops[1..] {
        let utry = op.get_utry(&params[index..index + op.num_params()], const_gates);
        result = result.kron(&permute_unitary(
            utry.view(),
            circuit_size,
            op.location.clone(),
        ));
        index += op.num_params();
    }
    (index, result)
}

/// A list of gates in a quantum circuit
#[derive(Clone)]
pub struct Circuit {
    pub size: usize,
    pub radixes: Vec<usize>,
    pub ops: Vec<Operation>,
    pub constant_gates: Vec<Array2<Complex64>>,
    pub cycle_boundaries: Vec<(usize, usize)>,
    pub num_params: usize,
    pub sendable: bool,
    pub backend: SimulationBackend,
}

impl Circuit {
    pub fn new(
        size: usize,
        radixes: Vec<usize>,
        ops_with_cycles: Vec<(Cycle, Operation)>,
        constant_gates: Vec<Array2<Complex64>>,
        backend: SimulationBackend,
    ) -> Self {
        let mut sendable = true;
        let mut num_params = 0;
        let mut cycle_boundaries = Vec::new();
        let mut current_cycle = 0usize;
        for (cycle, op) in &ops_with_cycles {
            num_params += op.gate.num_params();
            match op.gate {
                Gate::Dynamic(_) => sendable = false,
                _ => (),
            }
            if *cycle != current_cycle {
                cycle_boundaries.push((current_cycle, *cycle));
                current_cycle += 1;
            }
        }
        Circuit {
            size,
            radixes,
            ops: ops_with_cycles.iter().map(|(_, op)| op.clone()).collect(),
            constant_gates,
            cycle_boundaries,
            num_params,
            sendable,
            backend,
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

    fn get_utry(&self, params: &[f64], const_gates: &[Array2<Complex64>]) -> Array2<Complex64> {
        match &self.backend {
            SimulationBackend::Tensor => {
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
            SimulationBackend::Matrix => {
                let mut matrices = vec![];
                let mut locations = vec![];

                if params.is_empty() {
                    for op in &self.ops {
                        let mut utry = op.get_utry(&[], const_gates);
                        utry = permute_unitary(utry.view(), self.size, op.location.clone());
                        matrices.push(utry);
                        locations.push(&op.location);
                    }
                } else {
                    let mut param_idx = 0;
                    for op in &self.ops {
                        let mut utry = op.get_utry(
                            &params[param_idx..param_idx + op.num_params()],
                            const_gates,
                        );
                        param_idx += op.num_params();
                        utry = permute_unitary(utry.view(), self.size, op.location.clone());
                        matrices.push(utry);
                    }
                }

                matrices.iter().cloned().reduce(|a, b| b.matmul(a.view())).unwrap()
            }
        }
    }
}

impl Gradient for Circuit {
    fn get_utry_and_grad(
        &self,
        params: &[f64],
        const_gates: &[Array2<Complex64>],
    ) -> (Array2<Complex64>, Array3<Complex64>) {
        match &self.backend {
            SimulationBackend::Tensor => {
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
                    let perm_t = perm.t();
                    let id = Array2::eye(2usize.pow((self.size - location.len()) as u32));

                    right.apply_left(m.view(), location, true);
                    let right_utry = right.get_utry();
                    let left_utry = left.get_utry();
                    for grad in d_m.outer_iter() {
                        let mut full_grad = grad.kron(&id);
                        full_grad = perm.matmul(full_grad.view());
                        full_grad = full_grad.matmul(perm_t);
                        let right_grad = right_utry.matmul(full_grad.view());
                        full_grads.push(right_grad.matmul(left_utry.view()));
                    }
                    left.apply_right(m.view(), location, false);
                }

                for (mut arr, grad) in out_grad.outer_iter_mut().zip(full_grads) {
                    arr.assign(&grad);
                }

                (left.get_utry(), out_grad)
            }
            SimulationBackend::Matrix => {
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
                        matrices.push(permute_unitary(utry.view(), self.size, op.location.clone()));
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
                        matrices.push(permute_unitary(utry.view(), self.size, op.location.clone()));
                        grads.push(grad);
                        locations.push(&op.location);
                    }
                }

                let dim = 2usize.pow(self.size as u32);

                
                let mut left = Array2::eye(dim);
                let mut right = matrices.iter().cloned().reduce(|a, b| b.matmul(a.view())).unwrap();
                let mut full_grads = Vec::with_capacity(num_grads);
                let mut out_grad = Array3::zeros((
                    num_grads,
                    2usize.pow(self.size as u32),
                    2usize.pow(self.size as u32),
                ));

                for (m, location, d_m) in izip!(matrices, locations, grads) {
                    right = right.matmul(m.conj().t());
                    for grad in d_m.outer_iter() {
                        let full_grad = permute_unitary(grad.view(), self.size, location.clone());
                        let tmp = full_grad.matmul(left.view());
                        full_grads.push(right.matmul(tmp.view()));
                    }
                    left = m.matmul(left.view());
                }

                for (mut arr, grad) in out_grad.outer_iter_mut().zip(full_grads) {
                    arr.assign(&grad);
                }

                (left, out_grad)
            }
        }
    }

    fn get_grad(&self, params: &[f64], const_gates: &[Array2<Complex64>]) -> Array3<Complex64> {
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
