use num_complex::Complex64;
use numpy::PyArray2;
use pyo3::prelude::*;

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
        py: Python,
        mut circuit: Circuit,
        target: PyObject,
        x0: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let target_rs = match target.extract::<Py<PyArray2<Complex64>>>(py) {
            Ok(arr) => arr,
            Err(..) => {
                let target_np = target.getattr(py, "numpy")?;
                target_np.extract::<Py<PyArray2<Complex64>>>(py)?
            }
        };
        let target_rs = target_rs.as_ref(py).to_owned_array();
        Ok(self.instantiator.instantiate(&mut circuit, target_rs, &x0))
    }
}
