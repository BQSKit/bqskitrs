use crate::gates::Gradient;
use crate::gates::Optimize;
use crate::gates::Size;
use crate::gates::Unitary;
use crate::{i, r};

use ndarray::Array2;
use ndarray::Array3;
use num_complex::Complex64;

/// IBM's U3 single qubit gate
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct U3Gate();

impl U3Gate {
    pub fn new() -> Self {
        U3Gate {}
    }
}

impl Unitary for U3Gate {
    fn num_params(&self) -> usize {
        3
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<Complex64>]) -> Array2<Complex64> {
        let ct = r!((params[0] / 2.0).cos());
        let st = r!((params[0] / 2.0).sin());
        let cp = (params[1]).cos();
        let sp = (params[1]).sin();
        let cl = (params[2]).cos();
        let sl = (params[2]).sin();
        Array2::from_shape_vec(
            (2, 2),
            vec![
                ct,
                -st * (cl + i!(1.0) * sl),
                st * (cp + i!(1.0) * sp),
                ct * (cl * cp - sl * sp + i!(1.0) * cl * sp + i!(1.0) * sl * cp),
            ],
        )
        .unwrap()
    }
}

impl Gradient for U3Gate {
    fn get_grad(&self, params: &[f64], _const_gates: &[Array2<Complex64>]) -> Array3<Complex64> {
        let ct = r!((params[0] / 2.0).cos());
        let st = r!((params[0] / 2.0).sin());
        let cp = (params[1]).cos();
        let sp = (params[1]).sin();
        let cl = (params[2]).cos();
        let sl = (params[2]).sin();
        Array3::from_shape_vec(
            (3, 2, 2),
            vec![
                // param 0
                -0.5 * st,
                -0.5 * ct * (cl + i!(1.0) * sl),
                0.5 * ct * (cp + i!(1.0) * sp),
                -0.5 * st * (cl * cp - sl * sp + i!(1.0) * cl * sp + i!(1.0) * sl * cp),
                // param 1
                r!(0.0),
                r!(0.0),
                st * r!(2.0) / 2.0 * (-sp + i!(1.0) * cp),
                ct * r!(2.0) / 2.0 * (cl * -sp - sl * cp + i!(1.0) * cl * cp + i!(1.0) * sl * -sp),
                //param 2
                r!(0.0),
                -st * r!(2.0) / 2.0 * (-sl + i!(1.0) * cl),
                r!(0.0),
                ct * r!(2.0) / 2.0 * (-sl * cp - cl * sp + i!(1.0) * -sl * sp + i!(1.0) * cl * cp),
            ],
        )
        .unwrap()
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        _const_gates: &[Array2<Complex64>],
    ) -> (Array2<Complex64>, Array3<Complex64>) {
        let ct = r!((params[0] / 2.0).cos());
        let st = r!((params[0] / 2.0).sin());
        let cp = (params[1]).cos();
        let sp = (params[1]).sin();
        let cl = (params[2]).cos();
        let sl = (params[2]).sin();
        (
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    ct,
                    -st * (cl + i!(1.0) * sl),
                    st * (cp + i!(1.0) * sp),
                    ct * (cl * cp - sl * sp + i!(1.0) * cl * sp + i!(1.0) * sl * cp),
                ],
            )
            .unwrap(),
            Array3::from_shape_vec(
                (3, 2, 2),
                vec![
                    // param 0
                    -0.5 * st,
                    -0.5 * ct * (cl + i!(1.0) * sl),
                    0.5 * ct * (cp + i!(1.0) * sp),
                    -0.5 * st * (cl * cp - sl * sp + i!(1.0) * cl * sp + i!(1.0) * sl * cp),
                    // param 1
                    r!(0.0),
                    r!(0.0),
                    st * r!(2.0) / 2.0 * (-sp + i!(1.0) * cp),
                    ct * r!(2.0) / 2.0
                        * (cl * -sp - sl * cp + i!(1.0) * cl * cp + i!(1.0) * sl * -sp),
                    //param 2
                    r!(0.0),
                    -st * r!(2.0) / 2.0 * (-sl + i!(1.0) * cl),
                    r!(0.0),
                    ct * r!(2.0) / 2.0
                        * (-sl * cp - cl * sp + i!(1.0) * -sl * sp + i!(1.0) * cl * cp),
                ],
            )
            .unwrap(),
        )
    }
}

impl Size for U3Gate {
    fn num_qudits(&self) -> usize {
        1
    }
}

impl Optimize for U3Gate {}
