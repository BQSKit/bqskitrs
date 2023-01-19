use ndarray::Array2;
use ndarray::Array3;
use ndarray_linalg::c64;

use super::Gradient;
use super::Optimize;
use super::Size;
use super::Unitary;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ConstantGate {
    index: usize,
    size: usize,
}

impl ConstantGate {
    pub fn new(index: usize, size: usize) -> Self {
        ConstantGate { index, size }
    }
}

impl Size for ConstantGate {
    fn num_qudits(&self) -> usize {
        self.size
    }
}

impl Unitary for ConstantGate {
    fn num_params(&self) -> usize {
        0
    }

    fn get_utry(&self, _params: &[f64], const_gates: &[Array2<c64>]) -> Array2<c64> {
        const_gates[self.index].clone()
    }
}

impl Gradient for ConstantGate {
    fn get_grad(&self, _params: &[f64], _const_gates: &[Array2<c64>]) -> Array3<c64> {
        Array3::zeros((0, 0, 0))
    }

    fn get_utry_and_grad(
        &self,
        _params: &[f64],
        const_gates: &[Array2<c64>],
    ) -> (Array2<c64>, Array3<c64>) {
        (const_gates[self.index].clone(), Array3::zeros((0, 0, 0)))
    }
}

impl Optimize for ConstantGate {}
