use crate::ir::gates::{Gradient, Size};
use crate::ir::gates::{Optimize, Unitary};
use crate::{i, r};

use ndarray::{Array2, Array3, ArrayViewMut2};
use ndarray_linalg::c64;

/// A gate representing a controlled X rotation.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct CRXGate();

impl CRXGate {
    pub fn new() -> Self {
        CRXGate {}
    }
}

impl Unitary for CRXGate {
    fn num_params(&self) -> usize {
        1
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<c64>]) -> Array2<c64> {
        let cos = r!((params[0] / 2.).cos());
        let sin = i!(-(params[0] / 2.).sin());
        let zero = r!(0.0);
        let one = r!(1.0);
        Array2::from_shape_vec(
            (4, 4),
            vec![
                one, zero, zero, zero, zero, one, zero, zero, zero, zero, cos, sin, zero, zero,
                sin, cos,
            ],
        )
        .unwrap()
    }
}

impl Gradient for CRXGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[Array2<c64>]) -> Array3<c64> {
        let dcos = -1. * r!(params[0] / 2.).sin() / 2.;
        let dsin = i!(-(params[0] / 2.).cos() / 2.);
        let zero = r!(0.0);
        Array3::from_shape_vec(
            (1, 4, 4),
            vec![
                zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, dcos, dsin, zero, zero,
                dsin, dcos,
            ],
        )
        .unwrap()
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        _const_gates: &[Array2<c64>],
    ) -> (Array2<c64>, Array3<c64>) {
        let cos = r!((params[0] / 2.).cos());
        let sin = i!(-(params[0] / 2.).sin());
        let dcos = -1. * r!(params[0] / 2.).sin() / 2.;
        let dsin = i!(-(params[0] / 2.).cos() / 2.);
        let zero = r!(0.0);
        let one = r!(1.0);
        (
            Array2::from_shape_vec(
                (4, 4),
                vec![
                    one, zero, zero, zero, zero, one, zero, zero, zero, zero, cos, sin, zero, zero,
                    sin, cos,
                ],
            )
            .unwrap(),
            Array3::from_shape_vec(
                (1, 4, 4),
                vec![
                    zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, dcos, dsin, zero,
                    zero, dsin, dcos,
                ],
            )
            .unwrap(),
        )
    }
}

impl Size for CRXGate {
    fn num_qudits(&self) -> usize {
        2
    }
}

impl Optimize for CRXGate {
    fn optimize(&self, _env_matrix: ArrayViewMut2<c64>) -> Vec<f64> {
        unimplemented!()
    }
}
