use std::mem::MaybeUninit;

use ndarray::{Array2, ArrayBase, ArrayView2, Data, Ix2, LinalgScalar, OwnedRepr, Zip};
use num_complex::Complex64;

pub trait Kronecker {
    fn kron(&self, other: &Array2<Complex64>) -> Array2<Complex64>;
}

/// Kronecker product of 2D matrices.
///
/// The kronecker product of a LxN matrix A and a MxR matrix B is a (L*M)x(N*R)
/// matrix K formed by the block multiplication A_ij * B.
fn kron<'a, A, S1, S2>(a: &ArrayBase<S1, Ix2>, b: &'a ArrayBase<S2, Ix2>) -> ArrayBase<OwnedRepr<A>, Ix2>
where
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: LinalgScalar,
    A: std::ops::Mul<&'a ArrayBase<S2, Ix2>, Output = ArrayBase<OwnedRepr<A>, Ix2>>,
{
    let dimar = a.shape()[0];
    let dimac = a.shape()[1];
    let dimbr = b.shape()[0];
    let dimbc = b.shape()[1];
    let mut out: Array2<MaybeUninit<A>> = Array2::uninit((dimar * dimbr, dimac * dimbc));
    Zip::from(out.exact_chunks_mut((dimbr, dimbc)))
        .and(a)
        .for_each(|out, a| {
            (*a * b).assign_to(out);
        });
    unsafe { out.assume_init() }
}

impl Kronecker for Array2<Complex64> {
    fn kron(&self, other: &Array2<Complex64>) -> Array2<Complex64> {
        kron(self, other)
    }
}

impl Kronecker for ArrayView2<'_, Complex64> {
    fn kron(&self, other: &Array2<Complex64>) -> Array2<Complex64> {
        kron(self, other)
    }
}
