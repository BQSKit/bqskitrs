use crate::ir::gates::{Gradient, Size};
use crate::ir::gates::{Optimize, Unitary};
use crate::{i, r};

use ndarray::{Array2, Array3, ArrayViewMut2};
use ndarray_linalg::c64;

/// A gate representing an arbitrary rotation around the YY axis
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct RYYGate();

impl RYYGate {
    pub fn new() -> Self {
        RYYGate {}
    }
}

impl Unitary for RYYGate {
    fn num_params(&self) -> usize {
        1
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<c64>]) -> Array2<c64> {
        let cos = r!((params[0] / 2.).cos());
        let nsin = i!(-1.0) * (params[0] / 2.).sin();
        let psin = i!(1.0) * (params[0] / 2.).sin();
        let zero = r!(0.0);
        Array2::from_shape_vec(
            (4, 4),
            vec![
                cos, zero, zero, psin, zero, cos, nsin, zero, zero, nsin, cos, zero, psin, zero,
                zero, cos,
            ],
        )
        .unwrap()
    }
}

impl Gradient for RYYGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[Array2<c64>]) -> Array3<c64> {
        let dcos = -1. * r!(params[0] / 2.).sin() / 2.;
        let dnsin = i!(-1.0) * (params[0] / 2.).cos() / 2.;
        let dpsin = i!(1.0) * (params[0] / 2.).cos() / 2.;
        let zero = r!(0.0);
        Array3::from_shape_vec(
            (1, 4, 4),
            vec![
                dcos, zero, zero, dpsin, zero, dcos, dnsin, zero, zero, dnsin, dcos, zero, dpsin,
                zero, zero, dcos,
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
        let nsin = i!(-1.0) * (params[0] / 2.).sin();
        let psin = i!(1.0) * (params[0] / 2.).sin();
        let zero = r!(0.0);
        let dcos = -1. * r!(params[0] / 2.).sin() / 2.;
        let dnsin = i!(-1.0) * (params[0] / 2.).cos() / 2.;
        let dpsin = i!(1.0) * (params[0] / 2.).cos() / 2.;

        (
            Array2::from_shape_vec(
                (4, 4),
                vec![
                    cos, zero, zero, psin, zero, cos, nsin, zero, zero, nsin, cos, zero, psin,
                    zero, zero, cos,
                ],
            )
            .unwrap(),
            Array3::from_shape_vec(
                (1, 4, 4),
                vec![
                    dcos, zero, zero, dpsin, zero, dcos, dnsin, zero, zero, dnsin, dcos, zero,
                    dpsin, zero, zero, dcos,
                ],
            )
            .unwrap(),
        )
    }
}

impl Size for RYYGate {
    fn num_qudits(&self) -> usize {
        2
    }
}

impl Optimize for RYYGate {
    fn optimize(&self, _env_matrix: ArrayViewMut2<c64>) -> Vec<f64> {
        unimplemented!()
    }
}
