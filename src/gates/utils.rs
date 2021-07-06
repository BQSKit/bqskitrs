use ndarray::{Array2, Array3};
use num_complex::Complex64;

#[inline(always)]
pub fn rot_x(theta: f64) -> Array2<Complex64> {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
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
pub fn rot_x_jac(theta: f64) -> Array3<Complex64> {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    let half = Complex64::new(0.5, 0.0);
    let neghalf = Complex64::new(-0.5, 0.0);
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
pub fn rot_y(theta: f64) -> Array2<Complex64> {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
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
pub fn rot_y_jac(theta: f64) -> Array3<Complex64> {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let neghalf = Complex64::new(-0.5, 0.0);
    let half = Complex64::new(0.5, 0.0);
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
pub fn rot_z(theta: f64, phase: Option<Complex64>) -> Array2<Complex64> {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    let posi = Complex64::new(0.0, 1.0);
    let zero = Complex64::new(0.0, 0.0);
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
pub fn rot_z_jac(theta: f64, phase: Option<Complex64>) -> Array3<Complex64> {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    let posi = Complex64::new(0.0, 1.0);
    let zero = Complex64::new(0.0, 0.0);
    let half = Complex64::new(0.5, 0.0);
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
