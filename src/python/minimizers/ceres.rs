use pyo3::{exceptions::PyTypeError, prelude::*, types::PyTuple};

use crate::minimizers::{CeresJacSolver, Minimizer, ResidualFunction};

use numpy::IntoPyArray;
use numpy::PyArray1;

#[pyclass(name = "LeastSquaresMinimizerNative", subclass, module = "bqskitrs")]
pub struct PyCeresJacSolver {
    #[pyo3(get)]
    distance_metric: String,
    num_threads: usize,
    ftol: f64,
    gtol: f64,
    report: bool,
}

#[pymethods]
impl PyCeresJacSolver {
    #[new]
    #[args(num_threads = "1", ftol = "1e-6", gtol = "1e-10", report = "false")]
    fn new(num_threads: usize, ftol: f64, gtol: f64, report: bool) -> Self {
        println!("{:?}, {:?}, {:?}, {:?}", num_threads, ftol, gtol, report);
        Self {
            distance_metric: String::from("Residuals"),
            num_threads,
            ftol,
            gtol,
            report,
        }
    }

    fn minimize(&self, py: Python, cost_fn: PyObject, x0: PyObject) -> PyResult<Py<PyArray1<f64>>> {
        let x0_rust = x0.extract::<Vec<f64>>(py)?;
        let solv = CeresJacSolver::new(self.num_threads, self.ftol, self.gtol, self.report);
        let cost_fun = match cost_fn.extract::<ResidualFunction>(py) {
            Ok(fun) => Ok(fun),
            Err(err) => Err(PyTypeError::new_err(err.to_string())),
        }?;
        let x = if cost_fun.is_sendable() {
            py.allow_threads(move || solv.minimize(&cost_fun, &x0_rust))
        } else {
            solv.minimize(&cost_fun, &x0_rust)
        };
        Ok(x.into_pyarray(py).to_owned())
    }

    pub fn __reduce__(slf: PyRef<Self>) -> PyResult<(PyObject, PyObject)> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let num_threads = PyTuple::new(py, &[slf.num_threads]).into_py(py);
        let slf_ob: PyObject = slf.into_py(py);
        let cls = slf_ob.getattr(py, "__class__")?;
        Ok((cls, num_threads))
    }
}
