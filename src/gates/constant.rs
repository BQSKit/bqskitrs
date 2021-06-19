use super::Gradient;
use super::Optimize;
use super::Size;
use super::Unitary;
use squaremat::SquareMatrix;

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
    fn get_size(&self) -> usize {
        self.size
    }
}

impl Unitary for ConstantGate {
    fn num_params(&self) -> usize {
        0
    }

    fn get_utry(&self, _params: &[f64], const_gates: &[SquareMatrix]) -> SquareMatrix {
        const_gates[self.index].clone()
    }
}

impl Gradient for ConstantGate {
    fn get_grad(&self, _params: &[f64], _const_gates: &[SquareMatrix]) -> Vec<SquareMatrix> {
        vec![]
    }
}

impl Optimize for ConstantGate {}
