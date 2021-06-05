use num_complex::Complex64;
use pyo3::prelude::*;

use numpy::{PyArray1, PyArray2};

#[pyclass(name = "LBFGSMinimizerNative", module = "bqskitrs")]
struct PyBfgsJacSolver {
    size: usize,
    #[pyo3(get)]
    distance_metric: String,
}

#[pymethods]
impl PyBfgsJacSolver {
    #[new]
    /// Create a new L-BFGS Minimizer
    /// Args:
    ///   memoryize(int): The amount of memory to give L-BFGS in MB. 
    fn new(memory_size: Option<usize>) -> Self {
        if let Some(size) = memory_size {
            PyBfgsJacSolver {
                size,
                distance_metric: String::from("Frobenius"),
            }
        } else {
            PyBfgsJacSolver {
                size: 10,
                distance_metric: String::from("Frobenius"),
            }
        }
    }

    fn minimize(
        &self,
        py: Python,
        cost_fn: PyObject,
        x0: Option<PyObject>,
    ) -> PyResult<Py<PyArray1<f64>>> {

        let u = options.getattr(py, "target")?;
        let (circ, constant_gates) = match circuit.extract::<Py<PyGateWrapper>>(py) {
            Ok(c) => {
                let pygate = c.as_ref(py).try_borrow().unwrap();
                (pygate.gate.clone(), pygate.constant_gates.clone())
            }
            Err(_) => {
                let mut constant_gates = Vec::new();
                let gate = object_to_gate(&circuit, &mut constant_gates, py)?;
                (gate, constant_gates)
            }
        };
        let unitary =
            SquareMatrix::from_ndarray(u.extract::<&PySquareMatrix>(py)?.to_owned_array());
        let x0_rust = if let Some(x) = x0 {
            Some(x.extract::<Vec<f64>>(py)?)
        } else {
            None
        };
        let solv = BfgsJacSolver::new(self.size);
        let (mat, x0) = solv.solve_for_unitary(&circ, &constant_gates, &unitary, x0_rust);
        Ok((
            PySquareMatrix::from_array(py, &mat.into_ndarray()).to_owned(),
            PyArray1::from_vec(py, x0).to_owned(),
        ))
    }

    pub fn __reduce__(slf: PyRef<Self>) -> PyResult<(PyObject, PyObject)> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let slf_ob: PyObject = slf.into_py(py);
        let cls = slf_ob.getattr(py, "__class__")?;
        Ok((cls, PyTuple::empty(py).into_py(py)))
    }
}
