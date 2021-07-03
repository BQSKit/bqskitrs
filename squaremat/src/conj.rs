use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;

pub trait Conj {
    fn conj(&self) -> Array2<Complex64>;
}

impl Conj for Array2<Complex64> {
    fn conj(&self) -> Array2<Complex64> {
        self.mapv(|i| i.conj())
    }
}

impl Conj for ArrayView2<'_, Complex64> {
    fn conj(&self) -> Array2<Complex64> {
        self.mapv(|i| i.conj())
    }
}
