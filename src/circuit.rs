use crate::{gates::{Gate, Gradient, Unitary}, unitary_builder::UnitaryBuilder};

use ndarray::{Array2, ArrayD};
use num_complex::Complex64;
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
    pub fn new(size: usize, radixes: Vec<usize>, gates: Vec<(Gate, Location)>, constant_gates: Vec<SquareMatrix>) -> Self {
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
            if gate.num_params() != 0 {
                let utry = gate.get_utry(&params[param_idx..param_idx + gate.num_params()], const_gates);
                builder.apply_right(utry, location, false);
                param_idx += gate.num_params();
            } else {
                let arr = [];
                gate.get_utry(&arr, const_gates);
            }
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
        unimplemented!()
    }
}
