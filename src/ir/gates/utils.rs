use ndarray::{Array2, Array3};
use ndarray_linalg::c64;

#[inline(always)]
pub fn rot_x(theta: f64) -> Array2<c64> {
    let half_theta = c64::new(theta / 2.0, 0.0);
    let negi = c64::new(0.0, -1.0);
    unsafe {
        Array2::from_shape_vec_unchecked(
            (2, 2),
            vec![
                half_theta.cos(),
                negi * half_theta.sin(),
                negi * half_theta.sin(),
                half_theta.cos(),
            ],
        )
    }
}

#[inline(always)]
pub fn rot_x_jac(theta: f64) -> Array3<c64> {
    let half_theta = c64::new(theta / 2.0, 0.0);
    let negi = c64::new(0.0, -1.0);
    let half = c64::new(0.5, 0.0);
    let neghalf = c64::new(-0.5, 0.0);
    unsafe {
        Array3::from_shape_vec_unchecked(
            (1, 2, 2),
            vec![
                neghalf * half_theta.sin(),
                negi * half * half_theta.cos(),
                negi * half * half_theta.cos(),
                neghalf * half_theta.sin(),
            ],
        )
    }
}

#[inline(always)]
pub fn rot_y(theta: f64) -> Array2<c64> {
    let half_theta = c64::new(theta / 2.0, 0.0);
    unsafe {
        Array2::from_shape_vec_unchecked(
            (2, 2),
            vec![
                half_theta.cos(),
                -half_theta.sin(),
                half_theta.sin(),
                half_theta.cos(),
            ],
        )
    }
}

#[inline(always)]
pub fn rot_y_jac(theta: f64) -> Array3<c64> {
    let half_theta = c64::new(theta / 2.0, 0.0);
    let neghalf = c64::new(-0.5, 0.0);
    let half = c64::new(0.5, 0.0);
    unsafe {
        Array3::from_shape_vec_unchecked(
            (1, 2, 2),
            vec![
                neghalf * half_theta.sin(),
                neghalf * half_theta.cos(),
                half * half_theta.cos(),
                neghalf * half_theta.sin(),
            ],
        )
    }
}

#[inline(always)]
pub fn rot_z(theta: f64, phase: Option<c64>) -> Array2<c64> {
    let half_theta = c64::new(theta / 2.0, 0.0);
    let negi = c64::new(0.0, -1.0);
    let posi = c64::new(0.0, 1.0);
    let zero = c64::new(0.0, 0.0);
    if let Some(phase) = phase {
        unsafe {
            Array2::from_shape_vec_unchecked(
                (2, 2),
                vec![
                    phase * (negi * half_theta).exp(),
                    zero,
                    zero,
                    phase * (posi * half_theta).exp(),
                ],
            )
        }
    } else {
        unsafe {
            Array2::from_shape_vec_unchecked(
                (2, 2),
                vec![
                    (negi * half_theta).exp(),
                    zero,
                    zero,
                    (posi * half_theta).exp(),
                ],
            )
        }
    }
}

#[inline(always)]
pub fn rot_z_jac(theta: f64, phase: Option<c64>) -> Array3<c64> {
    let half_theta = c64::new(theta / 2.0, 0.0);
    let negi = c64::new(0.0, -1.0);
    let posi = c64::new(0.0, 1.0);
    let zero = c64::new(0.0, 0.0);
    let half = c64::new(0.5, 0.0);
    if let Some(phase) = phase {
        unsafe {
            Array3::from_shape_vec_unchecked(
                (1, 2, 2),
                vec![
                    phase * negi * half * (negi * half_theta).exp(),
                    zero,
                    zero,
                    phase * posi * half * (posi * half_theta).exp(),
                ],
            )
        }
    } else {
        unsafe {
            Array3::from_shape_vec_unchecked(
                (1, 2, 2),
                vec![
                    negi * half * (negi * half_theta).exp(),
                    zero,
                    zero,
                    posi * half * (posi * half_theta).exp(),
                ],
            )
        }
    }
}
