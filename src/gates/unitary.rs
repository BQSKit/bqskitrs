use enum_dispatch::enum_dispatch;
use squaremat::SquareMatrix;

/// Trait to calculate the unitary for a given gate.
#[enum_dispatch]
pub trait Unitary {
    fn num_params(&self) -> usize;
    fn get_utry(&self, params: &[f64], const_gates: &[SquareMatrix]) -> SquareMatrix;
}
