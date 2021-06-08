use ndarray::{s, Array2, Array4, Ix2};
use num_complex::Complex64;
use squaremat::SquareMatrix;

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

const fn num_bits<T>() -> usize {
    std::mem::size_of::<T>() * 8
}

pub fn log_2(x: usize) -> usize {
    num_bits::<usize>() - x.leading_zeros() as usize - 1
}

#[cfg(any(feature = "bfgs", feature = "python"))]
pub fn matrix_distance_squared(a: &SquareMatrix, b: &SquareMatrix) -> f64 {
    // 1 - np.abs(np.trace(np.dot(A,B.H))) / A.shape[0]
    // converted to
    // 1 - np.abs(np.sum(np.multiply(A,np.conj(B)))) / A.shape[0]
    let bc = b.conj();
    let mul = a.multiply(&bc);
    let sum = mul.sum();
    let norm = sum.norm();
    1f64 - norm / a.size as f64
}

#[cfg(feature = "python")]
pub fn matrix_distance(a: &SquareMatrix, b: &SquareMatrix) -> f64 {
    let dist_sq = matrix_distance_squared(a, b);
    dist_sq.abs().sqrt()
}

#[cfg(any(feature = "bfgs", feature = "python"))]
pub fn matrix_distance_squared_jac(
    u: &SquareMatrix,
    m: &SquareMatrix,
    j: Vec<SquareMatrix>,
) -> (f64, Vec<f64>) {
    let s = u.multiply(&m.conj()).sum();
    let dsq = 1f64 - s.norm() / u.size as f64;
    if s == r!(0.0) {
        return (dsq, vec![std::f64::INFINITY; j.len()]);
    }
    let jus: Vec<Complex64> = j.iter().map(|ji| u.multiply(&ji.conj()).sum()).collect();
    let jacs = jus
        .iter()
        .map(|jusi| -(jusi.re * s.re + jusi.im * s.im) * u.size as f64 / s.norm())
        .collect();
    (dsq, jacs)
}

#[cfg(any(feature = "ceres", feature = "python"))]
/// Calculates the residuals
pub fn matrix_residuals(a: &SquareMatrix, b: &SquareMatrix, i: &Array2<f64>) -> Vec<f64> {
    let m = b.matmul(&a.H());
    let (re, im) = m.split_complex();
    let r = re - i;
    r.iter().chain(im.iter()).copied().collect()
}

#[cfg(any(feature = "ceres", feature = "python"))]
pub fn matrix_residuals_jac(
    u: &SquareMatrix,
    _m: &SquareMatrix,
    jacs: &[SquareMatrix],
) -> Array2<f64> {
    let u_h = u.H();
    Array2::from_shape_vec(
        (jacs.len(), u.size * u.size * 2),
        jacs.iter().fold(Vec::new(), |mut acc, j| {
            let m = j.matmul(&u_h.clone());
            let (re, im) = m.split_complex();
            let row: Vec<f64> = re.iter().chain(im.iter()).copied().collect();
            acc.extend(row);
            acc
        }),
    )
    .unwrap()
    .t()
    .to_owned()
}