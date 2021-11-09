use std::mem::MaybeUninit;

use ndarray::{Array2, ArrayView2, Zip};
use num_complex::Complex64;

fn multiply(first: &ArrayView2<Complex64>, other: &ArrayView2<Complex64>) -> Array2<Complex64> {
    let mut out = Array2::uninit((first.shape()[0], first.shape()[1]));
    Zip::from(out.view_mut())
        .and(first)
        .and(other)
        .for_each(|out, &first, &other| {
            *out = MaybeUninit::new(first * other);
        });
    unsafe { out.assume_init() }
}

pub trait Multiply {
    fn multiply(&self, other: &ArrayView2<Complex64>) -> Array2<Complex64>;
}

impl Multiply for Array2<Complex64> {
    fn multiply(&self, other: &ArrayView2<Complex64>) -> Array2<Complex64> {
        multiply(&self.view(), other)
    }
}

impl Multiply for ArrayView2<'_, Complex64> {
    fn multiply(&self, other: &ArrayView2<Complex64>) -> Array2<Complex64> {
        multiply(&self, &other)
    }
}
