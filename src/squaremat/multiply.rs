use std::mem::MaybeUninit;

use ndarray::{Array2, ArrayView2, Zip};
use ndarray_linalg::c64;

fn multiply(first: &ArrayView2<c64>, other: &ArrayView2<c64>) -> Array2<c64> {
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
    fn multiply(&self, other: &ArrayView2<c64>) -> Array2<c64>;
}

impl Multiply for Array2<c64> {
    fn multiply(&self, other: &ArrayView2<c64>) -> Array2<c64> {
        multiply(&self.view(), other)
    }
}

impl Multiply for ArrayView2<'_, c64> {
    fn multiply(&self, other: &ArrayView2<c64>) -> Array2<c64> {
        multiply(&self, &other)
    }
}
