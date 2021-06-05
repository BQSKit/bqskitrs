use crate::gates::Gradient;
use crate::gates::Size;
use crate::gates::Unitary;
use crate::{i, r};

use num_complex::Complex64;
use squaremat::SquareMatrix;

/// IBM's U2 single qubit gate
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct U2Gate();

impl U2Gate {
    pub fn new() -> Self {
        U2Gate {}
    }
}

impl Unitary for U2Gate {
    fn num_params(&self) -> usize {
        2
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        let phase = r!(1.0) / r!(2.0f64.sqrt());
        let e1 = (i!(1.0) * params[1]).exp();
        let e2 = (i!(1.0) * params[0]).exp();
        let e3 = (i!(1.0) * (params[0] + params[1])).exp();
        SquareMatrix::from_vec(vec![phase, -phase * e1, phase * e2, phase * e3], 2)
    }
}

impl Gradient for U2Gate {
    fn get_grad(&self, params: &[f64], _const_gates: &[SquareMatrix]) -> Vec<SquareMatrix> {
        let phase = r!(1.0) / r!(2.0f64.sqrt());
        let e1 = (i!(1.0) * params[1]).exp();
        let e2 = (i!(1.0) * params[0]).exp();
        let e3 = (i!(1.0) * (params[0] + params[1])).exp();
        vec![
            SquareMatrix::from_vec(
                vec![r!(0.0), r!(0.0), phase * i!(1.0) * e2, phase * i!(1.0) * e3],
                2,
            ),
            SquareMatrix::from_vec(
                vec![
                    r!(0.0),
                    phase * i!(-1.0) * e1,
                    r!(0.0),
                    phase * i!(1.0) * e3,
                ],
                2,
            ),
        ]
    }
}

impl Size for U2Gate {
    fn get_size(&self) -> usize {
        1
    }
}
