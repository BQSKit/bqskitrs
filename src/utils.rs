use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView2, ArrayView3, Ix2};
use ndarray_einsum_beta::einsum;
use num_complex::Complex64;
use squaremat::*;

use crate::r;

use itertools::Itertools;

pub fn trace(arr: Array4<Complex64>) -> Array2<Complex64> {
    let mut out = Array2::<Complex64>::zeros((arr.shape()[2], arr.shape()[3]));
    for i in 0..arr.shape()[2] {
        for j in 0..arr.shape()[3] {
            out[(i, j)] = arr
                .slice(s![.., .., i, ..])
                .slice(s![.., .., j])
                .into_dimensionality::<Ix2>()
                .unwrap()
                .into_owned()
                .diag()
                .sum();
        }
    }
    out
}

pub fn argsort(v: Vec<usize>) -> Vec<usize> {
    v.iter()
        .enumerate()
        .sorted_by(|(_idx_a, a), (_idx_b, b)| a.cmp(b))
        .map(|(idx, _a)| idx)
        .collect()
}

pub fn matrix_distance_squared(a: ArrayView2<Complex64>, b: ArrayView2<Complex64>) -> f64 {
    // 1 - np.abs(np.trace(np.dot(A,B.H))) / A.shape[0]
    // converted to
    // 1 - np.abs(np.sum(np.multiply(A,np.conj(B)))) / A.shape[0]
    let prod = einsum("ij,ij->", &[&a, &b.conj()]).unwrap();
    let norm = prod.sum().norm();
    1f64 - norm / a.shape()[0] as f64
}

pub fn matrix_distance_squared_jac(
    u: ArrayView2<Complex64>,
    m: ArrayView2<Complex64>,
    j: ArrayView3<Complex64>,
) -> (f64, Vec<f64>) {
    let size = u.shape()[0];
    let s = u.multiply(m.conj().view()).sum();
    let dsq = 1f64 - s.norm() / size as f64;
    if s == r!(0.0) {
        return (dsq, vec![std::f64::INFINITY; j.len()]);
    }
    let jus: Vec<Complex64> = j
        .outer_iter()
        .map(|ji| einsum("ij,ij->", &[&u, &ji.conj()]).unwrap().sum())
        .collect();
    let jacs = jus
        .iter()
        .map(|jusi| -(jusi.re * s.re + jusi.im * s.im) * size as f64 / s.norm())
        .collect();
    (dsq, jacs)
}

/// Calculates the residuals
pub fn matrix_residuals(
    a_matrix: &Array2<Complex64>,
    b_matrix: &Array2<Complex64>,
    identity: &Array2<f64>,
) -> Vec<f64> {
    let calculated_mat = b_matrix.dot(&a_matrix.conj().reversed_axes());
    let (re, im) = calculated_mat.split_complex();
    let resid = re - identity;
    resid.iter().chain(im.iter()).copied().collect()
}

pub fn matrix_residuals_jac(
    u: &Array2<Complex64>,
    _m: &Array2<Complex64>,
    jacs: &Array3<Complex64>,
) -> Array2<f64> {
    let u_h = u.conj().reversed_axes();
    let size = u.shape()[0];
    let mut out = Array2::zeros((jacs.shape()[0], size * size * 2));
    for (jac, mut row) in jacs.outer_iter().zip(out.rows_mut()) {
        let m = jac.dot(&u_h);
        let (re, im) = m.split_complex();
        let data = Array1::from_vec(re.iter().chain(im.iter()).copied().collect());
        row.assign(&data);
    }
    out.reversed_axes()
}
