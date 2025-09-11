use ndarray::{Array2, Array3, ArrayViewMut2};
use ndarray_linalg::c64;
use ndarray_linalg::SVD;

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
pub fn svd(matrix: ArrayViewMut2<c64>) -> (Array2<c64>, Array2<c64>) {
    /// Return Unitaries from SVD decomposition
    let result = matrix.svd(true, true);
    let actual_result = match result {
        Ok(res)  => res,
        Err(_res) => panic!("Problem svding the matrix: {:?}", matrix),
    };

    // Safety: u/vt are the same size since matrix is a square matrix with sides of size `size`
    (actual_result.0.unwrap(), actual_result.2.unwrap())
}

#[inline(always)]
pub fn imag_exp(theta: f64) -> c64 {
    // Return e^(i * theta)
    let posi = c64::new(0.0, 1.0);
    (posi * theta).exp()
}

#[inline(always)]
pub fn get_indices(index: usize, target_qudit: usize, num_qudits: usize) -> (usize, usize) {
    // Get indices for the matrix based on the target qubit.
    let shift_qubit = num_qudits - target_qudit - 1;
    let shift = 2_usize.pow(shift_qubit as u32);
    
    // Split into two parts around target qubit
    // 100 | 111
    let left = index / shift;
    let right = index % shift;

    // Now, shift left by one spot to
    // make room for the target qubit
    let left_shifted = left * (shift * 2);
    
    // Now add 0 * new_ind and 1 * new_ind to get indices
    (left_shifted + right, left_shifted + shift + right)
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
