use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;

pub trait Trace {
    fn trace(&self) -> Complex64;
}

impl Trace for Array2<Complex64> {
    fn trace(&self) -> Complex64 {
        self.diag().sum()
    }
}

impl Trace for ArrayView2<'_, Complex64> {
    fn trace(&self) -> Complex64 {
        self.diag().sum()
    }
}
