use enum_dispatch::enum_dispatch;
use ndarray::{Array2, Array3};
use num_complex::Complex64;

use super::Unitary;
/// Gradient should be implemented for all gates where one can take their gradient.
#[enum_dispatch]
pub trait Gradient: Unitary {
    /// Get the gradient and unitary together
    fn get_utry_and_grad(
        &self,
        params: &[f64],
        const_gates: &[Array2<Complex64>],
    ) -> (Array2<Complex64>, Array3<Complex64>) {
        (
            self.get_utry(params, const_gates),
            self.get_grad(params, const_gates),
        )
    }

    /// Get the gradient of `self`.
    fn get_grad(&self, params: &[f64], const_gates: &[Array2<Complex64>]) -> Array3<Complex64>;
}
