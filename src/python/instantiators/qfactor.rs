use num_complex::Complex64;
use numpy::PyArray2;
use pyo3::prelude::*;
use squaremat::SquareMatrix;

use crate::{
    circuit::Circuit,
    instantiators::{Instantiate, QFactorInstantiator},
};

#[pyclass(name = "QFactorInstantiatorNative", subclass, module = "bqskitrs")]
pub struct PyQFactorInstantiator {
    instantiator: QFactorInstantiator,
}

#[pymethods]
impl PyQFactorInstantiator {
    #[new]
    /// Create a new QFactor Instantiator
    fn new(
        diff_tol_a: Option<f64>,
        diff_tol_r: Option<f64>,
        dist_tol: Option<f64>,
        max_iters: Option<usize>,
        min_iters: Option<usize>,
        slowdown_factor: Option<f64>,
        reinit_delay: Option<usize>,
    ) -> Self {
        Self {
            instantiator: QFactorInstantiator::new(
                diff_tol_a,
                diff_tol_r,
                dist_tol,
                max_iters,
                min_iters,
                slowdown_factor,
                reinit_delay,
            ),
        }
    }

    pub fn instantiate(
        &self,
        circuit: Circuit,
        target: &PyArray2<Complex64>,
        x0: Vec<f64>,
    ) -> Vec<f64> {
        let target_rs = SquareMatrix::from_ndarray(target.to_owned_array());
        self.instantiator.instantiate(circuit, target_rs, &x0)
    }
}
