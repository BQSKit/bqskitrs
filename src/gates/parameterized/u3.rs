use crate::gates::Gradient;
use crate::gates::Size;
use crate::gates::Unitary;
use crate::{i, r};

use num_complex::Complex64;
use squaremat::SquareMatrix;

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

    fn get_utry(&self, params: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        let ct = r!((params[0] / 2.0).cos());
        let st = r!((params[0] / 2.0).sin());
        let cp = (params[1]).cos();
        let sp = (params[1]).sin();
        let cl = (params[2]).cos();
        let sl = (params[2]).sin();
        SquareMatrix::from_vec(
            vec![
                ct,
                -st * (cl + i!(1.0) * sl),
                st * (cp + i!(1.0) * sp),
                ct * (cl * cp - sl * sp + i!(1.0) * cl * sp + i!(1.0) * sl * cp),
            ],
            2,
        )
    }
}

impl Gradient for U3Gate {
    fn get_grad(&self, params: &[f64], _const_gates: &[SquareMatrix]) -> Vec<SquareMatrix> {
        let ct = r!((params[0] / 2.0).cos());
        let st = r!((params[0] / 2.0).sin());
        let cp = (params[1]).cos();
        let sp = (params[1]).sin();
        let cl = (params[2]).cos();
        let sl = (params[2]).sin();
        vec![
            SquareMatrix::from_vec(
                vec![
                    -0.5 * st,
                    -0.5 * ct * (cl + i!(1.0) * sl),
                    0.5 * ct * (cp + i!(1.0) * sp),
                    -0.5 * st * (cl * cp - sl * sp + i!(1.0) * cl * sp + i!(1.0) * sl * cp),
                ],
                2,
            ),
            SquareMatrix::from_vec(
                vec![
                    r!(0.0),
                    r!(0.0),
                    st * r!(2.0) / 2.0 * (-sp + i!(1.0) * cp),
                    ct * r!(2.0) / 2.0
                        * (cl * -sp - sl * cp + i!(1.0) * cl * cp + i!(1.0) * sl * -sp),
                ],
                2,
            ),
            SquareMatrix::from_vec(
                vec![
                    r!(0.0),
                    -st * r!(2.0) / 2.0 * (-sl + i!(1.0) * cl),
                    r!(0.0),
                    ct * r!(2.0) / 2.0
                        * (-sl * cp - cl * sp + i!(1.0) * -sl * sp + i!(1.0) * cl * cp),
                ],
                2,
            ),
        ]
    }
}

impl Size for U3Gate {
    fn get_size(&self) -> usize {
        1
    }
}
