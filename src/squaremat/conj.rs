use ndarray::{Array2, ArrayView2};
use ndarray_linalg::c64;

pub trait Conj {
    fn conj(&self) -> Array2<c64>;
}

impl Conj for Array2<c64> {
    fn conj(&self) -> Array2<c64> {
        self.mapv(|i| i.conj())
    }
}

impl Conj for ArrayView2<'_, c64> {
    fn conj(&self) -> Array2<c64> {
        self.mapv(|i| i.conj())
    }
}
