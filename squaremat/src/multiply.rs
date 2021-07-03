use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;

pub trait Multiply {
    fn multiply(&self, other: ArrayView2<Complex64>) -> Array2<Complex64>;
}

impl Multiply for Array2<Complex64> {
    fn multiply(&self, other: ArrayView2<Complex64>) -> Array2<Complex64> {
        let size = self.shape()[0];
        // Safety: same size as `self`
        unsafe {
            Array2::from_shape_vec_unchecked(
                (size, size),
                self.iter().zip(other.iter()).map(|(a, b)| a * b).collect(),
            )
        }
    }
}

impl Multiply for ArrayView2<'_, Complex64> {
    fn multiply(&self, other: ArrayView2<Complex64>) -> Array2<Complex64> {
        let size = self.shape()[0];
        // Safety: same size as `self`
        unsafe {
            Array2::from_shape_vec_unchecked(
                (size, size),
                self.iter().zip(other.iter()).map(|(a, b)| a * b).collect(),
            )
        }
    }
}
