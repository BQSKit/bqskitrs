use crate::gates::Gate;
use crate::utils::log_2;
use crate::Circuit;

use ndarray::ShapeError;
use ndarray::{Array2, ArrayD};
use num_complex::Complex64;

/// A list of gates in a quantum circuit
pub struct TensorNetwork {
    circ: Circuit,
    tensor: ArrayD<Complex64>,
    target: Array2<Complex64>,
    num_qubits: usize,
}

impl TensorNetwork {
    pub fn new(circ: Circuit, tensor: ArrayD<Complex64>, target: Array2<Complex64>) -> Self {
        let num_qubits = log_2(target.shape()[0]);
        let mut slf = TensorNetwork {
            circ,
            tensor,
            target,
            num_qubits,
        };
        slf.initialize();
        slf
    }

    pub fn initialize(&mut self) -> Result<(), ShapeError> {
        self.tensor = self.target.t().map(|i| i.conj()).into_dyn();
        self.tensor = self
            .tensor
            .clone()
            .into_shape(vec![2; 2 * self.num_qubits])?;
        Ok(())
    }
}
