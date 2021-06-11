use crate::{
    gates::{Gate, Gradient, Unitary},
    permutation_matrix::calc_permutation_matrix,
    unitary_builder::UnitaryBuilder,
};

use squaremat::SquareMatrix;

type Location = Vec<usize>;

/// A list of gates in a quantum circuit
pub struct Circuit {
    size: usize,
    radixes: Vec<usize>,
    pub gates: Vec<(Gate, Location)>,
    pub constant_gates: Vec<SquareMatrix>,
}

impl Circuit {
    pub fn new(
        size: usize,
        radixes: Vec<usize>,
        gates: Vec<(Gate, Location)>,
        constant_gates: Vec<SquareMatrix>,
    ) -> Self {
        Circuit {
            size,
            radixes,
            gates,
            constant_gates,
        }
    }
}

impl Unitary for Circuit {
    fn num_params(&self) -> usize {
        self.gates.iter().map(|i| i.0.num_params()).sum()
    }

    fn get_utry(&self, params: &[f64], const_gates: &[SquareMatrix]) -> SquareMatrix {
        let mut param_idx = 0;
        let mut builder = UnitaryBuilder::new(self.size, self.radixes.clone());
        for (gate, location) in &self.gates {
            let utry = gate.get_utry(
                &params[param_idx..param_idx + gate.num_params()],
                const_gates,
            );
            param_idx += gate.num_params();
            builder.apply_right(&utry, location, false);
        }
        builder.get_utry()
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
        let mut param_idx = 0;
        for (gate, location) in &self.gates {
            let (utry, grad) = gate.get_utry_and_grad(
                &params[param_idx..param_idx + gate.num_params()],
                const_gates,
            );
            param_idx += gate.num_params();
            matrices.push(utry);
            grads.push(grad);
            locations.push(location);
        }

        let mut left = UnitaryBuilder::new(self.size, self.radixes.clone());
        let mut right = UnitaryBuilder::new(self.size, self.radixes.clone());
        let mut full_grads = vec![];
        for (M, location) in matrices.iter().zip(locations.iter()) {
            right.apply_right(M, location, false);
        }

        for ((M, location), dM) in matrices.iter().zip(locations.iter()).zip(grads) {
            let perm = calc_permutation_matrix(self.size, (**location).clone());
            let perm_t = perm.T();
            let id = SquareMatrix::eye(2usize.pow((self.size - location.len()) as u32));

            right.apply_left(M, location, true);
            let right_utry = right.get_utry();
            let left_utry = left.get_utry();
            for grad in dM {
                let mut full_grad = grad.kron(&id);
                full_grad = perm.matmul(&full_grad);
                full_grad = full_grad.matmul(&perm_t);
                let right_grad = right_utry.matmul(&full_grad);
                full_grads.push(right_grad.matmul(&left_utry));
            }
            left.apply_right(&M, location, false);
        }
        (left.get_utry(), full_grads)
    }
}
