use ndarray::ArrayViewMut2;
use num_complex::Complex64;
use squaremat::SquareMatrix;

use crate::gates::{Gate, Gradient, Optimize, Unitary};

#[derive(Clone)]
pub struct Operation {
    pub gate: Gate,
    pub location: Vec<usize>,
    pub params: Vec<f64>,
}

impl Operation {
    pub fn new(gate: Gate, location: Vec<usize>, params: Vec<f64>) -> Self {
        Self {
            gate,
            location,
            params,
        }
    }
}

impl Unitary for Operation {
    fn num_params(&self) -> usize {
        self.gate.num_params()
    }
    fn get_utry(&self, params: &[f64], const_gates: &[SquareMatrix]) -> SquareMatrix {
        if params.is_empty() {
            self.gate.get_utry(&self.params, const_gates)
        } else {
            self.gate.get_utry(params, const_gates)
        }
        
    }
}

impl Gradient for Operation {
    fn get_grad(&self, params: &[f64], const_gates: &[SquareMatrix]) -> Vec<SquareMatrix> {
        if params.is_empty() {
            self.gate.get_grad(&self.params, const_gates)
        } else {
            self.gate.get_grad(params, const_gates)
        }
    }

    fn get_utry_and_grad(&self, params: &[f64], const_gates: &[SquareMatrix]) -> (SquareMatrix, Vec<SquareMatrix>) {
        if params.is_empty() {
            self.gate.get_utry_and_grad(&self.params, const_gates)
        } else {
            self.gate.get_utry_and_grad(params, const_gates)
        }
    }
}

impl Optimize for Operation {
    fn optimize(&self, env_matrix: ArrayViewMut2<Complex64>) -> Vec<f64> {
        self.gate.optimize(env_matrix)
    }
}