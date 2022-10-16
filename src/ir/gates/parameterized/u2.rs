use crate::ir::gates::Gradient;
use crate::ir::gates::Optimize;
use crate::ir::gates::Size;
use crate::ir::gates::Unitary;
use crate::{i, r};

use ndarray::Array2;
use ndarray::Array3;
use ndarray_linalg::c64;

/// IBM's U2 single qubit gate
#[derive(Copy, Clone, Debug, PartialEq, Default)]
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

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<c64>]) -> Array2<c64> {
        let phase = r!(1.0) / r!(2.0f64.sqrt());
        let e1 = (i!(1.0) * params[1]).exp();
        let e2 = (i!(1.0) * params[0]).exp();
        let e3 = (i!(1.0) * (params[0] + params[1])).exp();
        Array2::from_shape_vec((2, 2), vec![phase, -phase * e1, phase * e2, phase * e3]).unwrap()
    }
}

impl Gradient for U2Gate {
    fn get_grad(&self, params: &[f64], _const_gates: &[Array2<c64>]) -> Array3<c64> {
        let phase = r!(1.0) / r!(2.0f64.sqrt());
        let e1 = (i!(1.0) * params[1]).exp();
        let e2 = (i!(1.0) * params[0]).exp();
        let e3 = (i!(1.0) * (params[0] + params[1])).exp();
        Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                // param 0
                r!(0.0),
                r!(0.0),
                phase * i!(1.0) * e2,
                phase * i!(1.0) * e3,
                // param 1
                r!(0.0),
                phase * i!(-1.0) * e1,
                r!(0.0),
                phase * i!(1.0) * e3,
            ],
        )
        .unwrap()
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        _const_gates: &[Array2<c64>],
    ) -> (Array2<c64>, Array3<c64>) {
        let phase = r!(1.0) / r!(2.0f64.sqrt());
        let e1 = (i!(1.0) * params[1]).exp();
        let e2 = (i!(1.0) * params[0]).exp();
        let e3 = (i!(1.0) * (params[0] + params[1])).exp();
        (
            Array2::from_shape_vec((2, 2), vec![phase, -phase * e1, phase * e2, phase * e3])
                .unwrap(),
            Array3::from_shape_vec(
                (2, 2, 2),
                vec![
                    // param 0
                    r!(0.0),
                    r!(0.0),
                    phase * i!(1.0) * e2,
                    phase * i!(1.0) * e3,
                    // param 1
                    r!(0.0),
                    phase * i!(-1.0) * e1,
                    r!(0.0),
                    phase * i!(1.0) * e3,
                ],
            )
            .unwrap(),
        )
    }
}

impl Size for U2Gate {
    fn num_qudits(&self) -> usize {
        1
    }
}

impl Optimize for U2Gate {}
