use enum_dispatch::enum_dispatch;
use ndarray::Array2;
use num_complex::Complex64;

/// Trait to calculate the unitary for a given gate.
#[enum_dispatch]
pub trait Unitary {
    fn num_params(&self) -> usize;
    fn get_utry(&self, params: &[f64], const_gates: &[Array2<Complex64>]) -> Array2<Complex64>;
}
