#[pyclass(name = "LeastSquares_Jac_SolverNative", module = "bqskitrs")]
struct PyCeresJacSolver {
    #[pyo3(get)]
    distance_metric: String,
    num_threads: usize,
    ftol: f64,
    gtol: f64,
}

#[pymethods]
impl PyCeresJacSolver {
    #[new]
    fn new(num_threads: Option<usize>, ftol: Option<f64>, gtol: Option<f64>) -> Self {
        let threads = if let Some(threads) = num_threads {
            threads
        } else {
            1
        };
        let ftol = if let Some(ftol) = ftol {
            ftol
        } else {
            1e-6 // Ceres documented default
        };
        let gtol = if let Some(gtol) = gtol {
            gtol
        } else {
            1e-10 // Ceres documented default
        };
        Self {
            distance_metric: String::from("Residuals"),
            num_threads: threads,
            ftol,
            gtol,
        }
    }

    fn solve_for_unitary(
        &self,
        py: Python,
        circuit: PyObject,
        options: PyObject,
        x0: Option<PyObject>,
    ) -> PyResult<(Py<PySquareMatrix>, Py<PyArray1<f64>>)> {
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
        let solv = LeastSquaresJacSolver::new(self.num_threads, self.ftol, self.gtol);
        let (mat, x0) = solv.solve_for_unitary(&circ, &constant_gates, &unitary, x0_rust);
        Ok((
            PySquareMatrix::from_array(py, &mat.into_ndarray()).to_owned(),
            PyArray1::from_vec(py, x0).to_owned(),
        ))
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
