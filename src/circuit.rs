use crate::gates::{Gate, Gradient, Unitary};

use ndarray::{Array2, ArrayD};
use num_complex::Complex64;
use squaremat::SquareMatrix;

/// A list of gates in a quantum circuit
pub struct Circuit {
    pub gates: Vec<Gate>,
    pub constant_gates: Vec<SquareMatrix>,
}

impl Circuit {
    pub fn new(gates: Vec<Gate>, constant_gates: Vec<SquareMatrix>) -> Self {
        Circuit {
            gates,
            constant_gates,
        }
    }
}

impl Unitary for Circuit {
    fn num_params(&self) -> usize {
        self.gates.iter().map(|i| i.num_params()).sum()
    }

    fn get_utry(&self, params: &[f64], const_gates: &[SquareMatrix]) -> SquareMatrix {
        unimplemented!()
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
        unimplemented!()
    }
}
