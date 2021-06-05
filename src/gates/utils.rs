use num_complex::Complex64;
use squaremat::SquareMatrix;

pub fn rot_x(theta: f64) -> SquareMatrix {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    SquareMatrix::from_vec(
        vec![
            half_theta.cos(),
            negi * half_theta.sin(),
            negi * half_theta.sin(),
            half_theta.cos(),
        ],
        2,
    )
}

pub fn rot_x_jac(theta: f64) -> SquareMatrix {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    let half = Complex64::new(0.5, 0.0);
    let neghalf = Complex64::new(-0.5, 0.0);
    SquareMatrix::from_vec(
        vec![
            neghalf * half_theta.sin(),
            negi * half * half_theta.cos(),
            negi * half * half_theta.cos(),
            neghalf * half_theta.sin(),
        ],
        2,
    )
}

pub fn rot_y(theta: f64) -> SquareMatrix {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    SquareMatrix::from_vec(
        vec![
            half_theta.cos(),
            -half_theta.sin(),
            half_theta.sin(),
            half_theta.cos(),
        ],
        2,
    )
}

pub fn rot_y_jac(theta: f64) -> SquareMatrix {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let neghalf = Complex64::new(-0.5, 0.0);
    let half = Complex64::new(0.5, 0.0);
    SquareMatrix::from_vec(
        vec![
            neghalf * half_theta.sin(),
            neghalf * half_theta.cos(),
            half * half_theta.cos(),
            neghalf * half_theta.sin(),
        ],
        2,
    )
}

pub fn rot_z(theta: f64) -> SquareMatrix {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    let posi = Complex64::new(0.0, 1.0);
    let zero = Complex64::new(0.0, 0.0);
    SquareMatrix::from_vec(
        vec![
            (negi * half_theta).exp(),
            zero,
            zero,
            (posi * half_theta).exp(),
        ],
        2,
    )
}

pub fn rot_z_jac(theta: f64) -> SquareMatrix {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    let posi = Complex64::new(0.0, 1.0);
    let zero = Complex64::new(0.0, 0.0);
    let half = Complex64::new(0.5, 0.0);
    SquareMatrix::from_vec(
        vec![
            negi * half * (negi * half_theta).exp(),
            zero,
            zero,
            posi * half * (posi * half_theta).exp(),
        ],
        2,
    )
}

pub fn rot_z_jac_mul(theta: f64, multiplier: Option<f64>) -> SquareMatrix {
    let half_theta = Complex64::new(theta / 2.0, 0.0);
    let negi = Complex64::new(0.0, -1.0);
    let posi = Complex64::new(0.0, 1.0);
    let zero = Complex64::new(0.0, 0.0);
    let mult = if let Some(mult) = multiplier {
        mult
    } else {
        1.0
    };
    SquareMatrix::from_vec(
        vec![
            mult * 0.5 * (-half_theta.sin() + negi * half_theta.cos()),
            zero,
            zero,
            mult * 0.5 * (-half_theta.sin() + posi * half_theta.cos()),
        ],
        2,
    )
}
