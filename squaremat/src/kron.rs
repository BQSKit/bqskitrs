use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use num_complex::Complex64;

pub trait Kronecker {
    fn kron(&self, other: &Array2<Complex64>) -> Array2<Complex64>;
}

impl Kronecker for Array2<Complex64> {
    fn kron(&self, other: &Array2<Complex64>) -> Array2<Complex64> {
        let row_a = self.shape()[0];
        let row_b = other.shape()[0];
        // Safety: out is initialized below.
        let mut out = Array2::uninit((row_a * row_b, row_a * row_b));
        for (mut chunk, elem) in out
            .exact_chunks_mut((row_b, row_b))
            .into_iter()
            .zip(self.iter())
        {
            let v = Array2::from_elem((row_b, row_b), *(elem)) * other;
            // safety: the next line assigns values to `chunk_assign`
            let chunk_assign: &mut ArrayViewMut2<Complex64> =
                unsafe { std::mem::transmute(&mut chunk) };
            chunk_assign.assign(&v);
        }
        unsafe { out.assume_init() }
    }
}

impl Kronecker for ArrayView2<'_, Complex64> {
    fn kron(&self, other: &Array2<Complex64>) -> Array2<Complex64> {
        let row_a = self.shape()[0];
        let row_b = other.shape()[0];
        // Safety: out is initialized below.
        let mut out = Array2::uninit((row_a * row_b, row_a * row_b));
        for (mut chunk, elem) in out
            .exact_chunks_mut((row_b, row_b))
            .into_iter()
            .zip(self.iter())
        {
            let v = Array2::from_elem((row_b, row_b), *(elem)) * other;
            // safety: the next line assigns values to `chunk_assign`
            let chunk_assign: &mut ArrayViewMut2<Complex64> =
                unsafe { std::mem::transmute(&mut chunk) };
            chunk_assign.assign(&v);
        }
        unsafe { out.assume_init() }
    }
}
