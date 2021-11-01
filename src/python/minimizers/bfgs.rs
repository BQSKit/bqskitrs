use pyo3::prelude::*;

use numpy::IntoPyArray;
use numpy::PyArray1;
use pyo3::types::PyTuple;

use crate::minimizers::BfgsJacSolver;
use crate::minimizers::CostFunction;
use crate::minimizers::Minimizer;

use pyo3::exceptions::PyTypeError;

#[pyclass(name = "LBFGSMinimizerNative", subclass, module = "bqskitrs")]
pub struct PyBfgsJacSolver {
    size: usize,
}

#[pymethods]
impl PyBfgsJacSolver {
    #[new]
    /// Create a new L-BFGS Minimizer
    /// Args:
    ///   memorysize(int): The amount of memory to give L-BFGS in MB.
    fn new(memory_size: Option<usize>) -> Self {
        if let Some(size) = memory_size {
            PyBfgsJacSolver { size }
        } else {
            PyBfgsJacSolver { size: 10 }
        }
    }

    fn minimize(&self, py: Python, cost_fn: PyObject, x0: PyObject) -> PyResult<Py<PyArray1<f64>>> {
        let x0_rust = x0.extract::<Vec<f64>>(py)?;
        let solv = BfgsJacSolver::new(self.size);
        let cost_fun = match cost_fn.extract::<CostFunction>(py) {
            Ok(fun) => Ok(fun),
            Err(err) => Err(PyTypeError::new_err(err.to_string())),
        }?;
        let x = solv.minimize(cost_fun, x0_rust);
        Ok(x.into_pyarray(py).to_owned())
    }

    pub fn __reduce__(slf: PyRef<Self>) -> PyResult<(PyObject, PyObject)> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let slf_ob: PyObject = slf.into_py(py);
        let cls = slf_ob.getattr(py, "__class__")?;
        Ok((cls, PyTuple::empty(py).into_py(py)))
    }
}
