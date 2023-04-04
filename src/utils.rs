use ndarray::ArrayView1;
use ndarray::{s, Array1, Array2, Array3, ArrayView2, ArrayView3, ArrayView4, Ix2};
use ndarray_einsum_beta::einsum;
use ndarray_linalg::{c64, Scalar};
use crate::squaremat::*;
use crate::r;

use itertools::Itertools;

pub fn trace(arr: ArrayView4<c64>) -> Array2<c64> {
    let mut out = Array2::<c64>::zeros((arr.shape()[2], arr.shape()[3]));
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

pub fn state_dot(a: ArrayView1<c64>, b: ArrayView1<c64>) -> c64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x.conj() * y).sum::<c64>()
}

pub fn state_infidelity(a: ArrayView1<c64>, b: ArrayView1<c64>) -> f64 {
    1.0 - state_dot(a, b).norm().powi(2)
}

pub fn state_residuals(a: ArrayView1<c64>, b: ArrayView1<c64>) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x-y).norm().square()).collect()
}

pub fn state_infidelity_jac(u: ArrayView1<c64>, m: ArrayView1<c64>, j: ArrayView2<c64>) -> (f64, Vec<f64>) {
    let d = state_dot(u, m);
    let infidelity = 1.0 - d.norm().powi(2);
    let d_infidelity = j.outer_iter().map(|dv| {let dd = state_dot(u, dv); -2.0 * (d.re * dd.re + d.im * dd.im)}).collect();
    (infidelity, d_infidelity)
}

pub fn state_residuals_jac(u: ArrayView1<c64>, m: ArrayView1<c64>, j: ArrayView2<c64>) -> Array2<f64> {
    let d: Vec<c64> = u.iter().zip(m.iter()).map(|(&x, &y)| x - y).collect();
    let mut out = Array2::zeros((j.shape()[0], u.shape()[0]));

    for (jac, mut row) in j.outer_iter().zip(out.rows_mut()) {
        let data = Array1::from_vec(d.iter().zip(jac.iter()).map(|(&x, &y)| -2.0 * (x.re*y.re + x.im*y.im)).collect());
        row.assign(&data);
    }
    out.reversed_axes()
}


pub fn matrix_distance_squared(a: ArrayView2<c64>, b: ArrayView2<c64>) -> f64 {
    // 1 - np.abs(np.trace(np.dot(A,B.H))) / A.shape[0]
    // converted to
    // 1 - np.abs(np.sum(np.multiply(A,np.conj(B)))) / A.shape[0]
    let prod = einsum("ij,ij->", &[&a, &b.conj()]).unwrap();
    let norm = prod.sum().norm();
    1f64 - norm / a.shape()[0] as f64
}

pub fn matrix_distance_squared_jac(
    u: ArrayView2<c64>,
    m: ArrayView2<c64>,
    j: ArrayView3<c64>,
) -> (f64, Vec<f64>) {
    let size = u.shape()[0];
    let s = u.multiply(&m.conj().view()).sum();
    let dsq = 1f64 - s.norm() / size as f64;
    if s.norm() == 0.0 {
        return (dsq, vec![std::f64::INFINITY; j.shape()[0]]);
    }
    let jus: Vec<c64> = j
        .outer_iter()
        .map(|ji| einsum("ij,ij->", &[&u, &ji.conj()]).unwrap().sum())
        .collect();
    let jacs = jus
        .iter()
        .map(|jusi| -(jusi.re * s.re + jusi.im * s.im) / (size as f64 * s.norm()))
        .collect();
    (dsq, jacs)
}

/// Calculates the residuals
pub fn matrix_residuals(
    a_matrix: &Array2<c64>,
    b_matrix: &Array2<c64>,
    identity: &Array2<f64>,
) -> Vec<f64> {
    let calculated_mat = b_matrix.matmul(a_matrix.conj().t());
    let (re, im) = calculated_mat.split_complex();
    let resid = re - identity;
    resid.iter().chain(im.iter()).copied().collect()
}

pub fn matrix_residuals_jac(
    u: &Array2<c64>,
    _m: &Array2<c64>,
    jacs: &Array3<c64>,
) -> Array2<f64> {
    let u_conj = u.conj();
    let size = u.shape()[0];
    let mut out = Array2::zeros((jacs.shape()[0], size * size * 2));
    for (jac, mut row) in jacs.outer_iter().zip(out.rows_mut()) {
        let m = jac.matmul(u_conj.t());
        let (re, im) = m.split_complex();
        let data = Array1::from_vec(re.iter().chain(im.iter()).copied().collect());
        row.assign(&data);
    }
    out.reversed_axes()
}