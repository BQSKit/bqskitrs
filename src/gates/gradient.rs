use enum_dispatch::enum_dispatch;

use squaremat::SquareMatrix;

use super::Unitary;
/// Gradient should be implemented for all gates where one can take their gradient.
#[enum_dispatch]
pub trait Gradient: Unitary {
    /// Get the gradient and unitary together
    fn get_utry_and_grad(
        &self,
        params: &[f64],
        const_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        (
            self.get_utry(params, const_gates),
            self.get_grad(params, const_gates),
        )
    }

    /// Get the gradient of `self`.
    fn get_grad(&self, params: &[f64], const_gates: &[SquareMatrix]) -> Vec<SquareMatrix>;
}
