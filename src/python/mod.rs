use num_complex::Complex64;

use numpy::{PyArray1, PyArray2};

use pyo3::class::basic::PyObjectProtocol;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::wrap_pyfunction;

use better_panic::install;

use squaremat::SquareMatrix;

use crate::gates::*;
use crate::utils::{
    matrix_distance_squared, matrix_distance_squared_jac, matrix_residuals, matrix_residuals_jac,
};

//#[cfg(any(feature = "ceres", feature = "bfgs"))]
//use crate::solvers::{BfgsJacSolver, CeresJacSolver, Solver};

//#[cfg(any(feature = "ceres", feature = "bfgs"))]
//mod solvers;

pub type PySquareMatrix = PyArray2<Complex64>;

#[pyclass(name = "Gate", module = "bqskitrs")]
struct PyGateWrapper {
    #[pyo3(get)]
    dits: u8,
    pub gate: Gate,
    constant_gates: Vec<SquareMatrix>,
}

#[pymethods]
impl PyGateWrapper {
    #[new]
    pub fn new(pygate: PyObject, py: Python) -> Self {
        let mut constant_gates = Vec::new();
        let gate = object_to_gate(&pygate, &mut constant_gates, py).unwrap();
        PyGateWrapper {
            dits: gate.dits(),
            gate: gate,
            constant_gates,
        }
    }

    pub fn get_grad(&mut self, py: Python, v: &PyArray1<f64>) -> Vec<Py<PySquareMatrix>> {
        let jac = self
            .gate
            .get_grad(unsafe { v.as_slice().unwrap() }, &mut self.constant_gates);
        jac.iter()
            .map(|j| PySquareMatrix::from_array(py, &j.clone().into_ndarray()).to_owned())
            .collect()
    }

    pub fn matrix(&mut self, py: Python, v: &PyArray1<f64>) -> Py<PySquareMatrix> {
        PySquareMatrix::from_array(
            py,
            &self
                .gate
                .get_utry(unsafe { v.as_slice().unwrap() }, &mut self.constant_gates)
                .into_ndarray(),
        )
        .to_owned()
    }

    #[getter]
    pub fn num_params(&self) -> usize {
        self.gate.num_params()
    }

    fn kind(&self) -> String {
        match self.gate {
            Gate::U3(..) => String::from("U3"),
            Gate::U2(..) => String::from("U2"),
            Gate::U1(..) => String::from("U1"),
            Gate::RX(..) => String::from("RX"),
            Gate::RY(..) => String::from("RY"),
            Gate::RZ(..) => String::from("RZ"),
            //Gate::XZXZ(..) => String::from("XZXZ"),
            Gate::Constant(..) => String::from("ConstantUnitary"),
            //Gate::SingleQutrit(..) => String::from("SingleQutrit"),
        }
    }

    /* pub fn __reduce__(slf: PyRef<Self>) -> PyResult<(PyObject, PyObject)> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let tup: (PyObject,) = (slf.as_python(py)?,);
        let slf_ob: PyObject = slf.into_py(py);
        let cls = slf_ob.getattr(py, "__class__")?;
        Ok((cls, tup.into_py(py)))
    } */
}

#[pyproto]
impl<'a> PyObjectProtocol<'a> for PyGateWrapper {
    fn __str__(&self) -> PyResult<String> {
        Ok(self.kind())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("RustGate()"))
    }

    fn __hash__(&self) -> PyResult<isize> {
        let digest = md5::compute(format!("{:?}", self.gate).as_bytes());
        Ok(digest.iter().enumerate().fold(0, |acc, (i, j)| {
            acc + *j as isize * (256isize).pow(i as u32)
        }))
    }
}

/* fn gate_to_object(
    gate: &Gate,
    py: Python,
    constant_gates: &[SquareMatrix],
    gates: &PyModule,
) -> PyResult<PyObject> {
    Ok(match gate {
        Gate::Identity(id) => {
            let gate: PyObject = gates.get("IdentityGate")?.extract()?;
            let args = PyTuple::new(
                py,
                vec![log_2(constant_gates[id.index].size), id.data.dits as usize],
            );
            gate.call1(py, args)?
        }
        Gate::U3(..) => {
            let gate: PyObject = gates.get("U3Gate")?.extract()?;
            gate.call0(py)?
        }
        Gate::U2(..) => {
            let gate: PyObject = gates.get("U2Gate")?.extract()?;
            gate.call0(py)?
        }
        Gate::U1(..) => {
            let gate: PyObject = gates.get("U1Gate")?.extract()?;
            gate.call0(py)?
        }
        Gate::X(..) => {
            let gate: PyObject = gates.get("XGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::Y(..) => {
            let gate: PyObject = gates.get("YGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::Z(..) => {
            let gate: PyObject = gates.get("ZGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::XZXZ(..) => {
            let gate: PyObject = gates.get("XZXZGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::Kronecker(kron) => {
            let gate: PyObject = gates.get("KroneckerGate")?.extract()?;
            let steps: Vec<PyObject> = kron
                .substeps
                .iter()
                .map(|i| gate_to_object(i, py, &constant_gates, gates).unwrap())
                .collect();
            let substeps = PyTuple::new(py, steps);
            gate.call1(py, substeps)?
        }
        Gate::Product(prod) => {
            let gate: PyObject = gates.get("ProductGate")?.extract()?;
            let steps: Vec<PyObject> = prod
                .substeps
                .iter()
                .map(|i| gate_to_object(i, py, &constant_gates, gates).unwrap())
                .collect();
            let substeps = PyTuple::new(py, steps);
            gate.call1(py, substeps)?
        }
        Gate::SingleQutrit(..) => {
            let gate: PyObject = gates.get("SingleQutritGate")?.extract()?;
            gate.call0(py)?
        }
        Gate::ConstantUnitary(u) => {
            let mat = constant_gates[u.index].clone();
            let gate: PyObject = gates.get("UGate")?.extract()?;
            let tup = PyTuple::new(
                py,
                [PySquareMatrix::from_array(py, &mat.into_ndarray()).to_owned()].iter(),
            );
            gate.call1(py, tup)?
        }
    })
}

fn object_to_gate(
    obj: &PyObject,
    constant_gates: &mut Vec<SquareMatrix>,
    py: Python,
) -> PyResult<Gate> {
    let cls = obj.getattr(py, "__class__")?;
    let dunder_name = cls.getattr(py, "__name__")?;
    let name: &str = dunder_name.extract(py)?;
    match name {
        "CNOTGate" => {
            let one = r!(1.0);
            let nil = r!(0.0);
            let index = constant_gates.len();
            constant_gates.push(SquareMatrix::from_vec(
                vec![
                    one, nil, nil, nil, nil, one, nil, nil, nil, nil, nil, one, nil, nil, one, nil,
                ],
                4,
            ));
            Ok(GateCNOT::new(index).into())
        }
        "IdentityGate" => {
            let index = constant_gates.len();
            let n = obj.getattr(py, "size")?.extract(py)?;
            let radixes = obj.getattr(py, "radixes")?.extract(py)?.iter();
            let first = radixes.next();
            let all_equal = radixes.all(|i| i == first);
            if !all_equal {
                return Err(exceptions::PyValueError::new_err(
                    "Identity has differing radixes, which is not supported by native code",
                ));
            }
            constant_gates.push(SquareMatrix::eye(2usize.pow(n)));
            Ok(GateIdentity::new(index).into())
        }
        "U3Gate" => Ok(GateU3::new().into()),
        "U2Gate" => Ok(GateU2::new().into()),
        "U1Gate" => Ok(GateU1::new().into()),
        "XGate" => Ok(GateX::new().into()),
        "YGate" => Ok(GateY::new().into()),
        "ZGate" => Ok(GateZ::new().into()),
        "XZXZGate" => {
            let index = constant_gates.len();
            constant_gates.push(crate::utils::rot_x(std::f64::consts::PI / 2.0));
            Ok(GateXZXZ::new(index).into())
        }
        "U8Gate" => Ok(GateSingleQutrit::new().into()),
        "Gate" => {
            let g = obj.extract::<Py<PyGateWrapper>>(py)?;
            let wrapper = g.as_ref(py).try_borrow()?;
            Ok(wrapper.gate.clone())
        }
        _ => {
            if obj.getattr(py, "num_inputs")?.extract::<usize>(py)? == 0 {
                let dits = obj.getattr(py, "qudits")?.extract::<u8>(py)?;
                let args: Vec<u8> = vec![];
                let pyobj = obj.call_method(py, "matrix", (args,), None)?;
                let pymat = pyobj.extract::<&PyArray2<Complex64>>(py)?;
                let mat = unsafe { pymat.as_array() };
                let index = constant_gates.len();
                constant_gates.push(SquareMatrix::from_ndarray(mat.to_owned()).T());
                Ok(GateConstantUnitary::new(index, dits).into())
            } else {
                Err(exceptions::PyValueError::new_err(format!(
                    "Unknown gate {}",
                    name
                )))
            }
        }
    }
}

#[pyfunction]
fn native_from_object(obj: PyObject, py: Python) -> PyResult<Py<PyGateWrapper>> {
    let mut constant_gates = Vec::new();
    let gate = object_to_gate(&obj, &mut constant_gates, py)?;
    Py::new(
        py,
        PyGateWrapper {
            dits: gate.dits(),
            gate,
            constant_gates,
        },
    )
}
*/
#[pymodule]
fn bqskitrs(_py: Python, m: &PyModule) -> PyResult<()> {
    install();
    #[pyfn(m, "matrix_distance_squared")]
    fn matrix_distance_squared_py(a: &PySquareMatrix, b: &PySquareMatrix) -> f64 {
        matrix_distance_squared(
            &SquareMatrix::from_ndarray(a.to_owned_array()),
            &SquareMatrix::from_ndarray(b.to_owned_array()),
        )
    }
    #[pyfn(m, "matrix_distance_squared_jac")]
    fn matrix_distance_squared_jac_py(
        a: &PySquareMatrix,
        b: &PySquareMatrix,
        jacs: Vec<&PySquareMatrix>,
    ) -> (f64, Vec<f64>) {
        matrix_distance_squared_jac(
            &SquareMatrix::from_ndarray(a.to_owned_array()),
            &SquareMatrix::from_ndarray(b.to_owned_array()),
            jacs.iter()
                .map(|j| SquareMatrix::from_ndarray(j.to_owned_array()))
                .collect(),
        )
    }
    #[pyfn(m, "matrix_residuals")]
    fn matrix_residuals_py(
        a: &PySquareMatrix,
        b: &PySquareMatrix,
        eye: &PyArray2<f64>,
    ) -> Vec<f64> {
        matrix_residuals(
            &SquareMatrix::from_ndarray(a.to_owned_array()),
            &SquareMatrix::from_ndarray(b.to_owned_array()),
            &eye.to_owned_array(),
        )
    }
    #[pyfn(m, "matrix_residuals_jac")]
    fn matrix_residuals_jac_py(
        py: Python,
        u: &PySquareMatrix,
        m: &PySquareMatrix,
        jacs: Vec<&PySquareMatrix>,
    ) -> Py<PyArray2<f64>> {
        let v: Vec<SquareMatrix> = jacs
            .iter()
            .map(|i| SquareMatrix::from_ndarray(i.to_owned_array()))
            .collect();
        PyArray2::from_array(
            py,
            &matrix_residuals_jac(
                &SquareMatrix::from_ndarray(u.to_owned_array()),
                &SquareMatrix::from_ndarray(m.to_owned_array()),
                &v,
            ),
        )
        .to_owned()
    }
    //m.add_wrapped(wrap_pyfunction!(native_from_object))?;
    m.add_class::<PyGateWrapper>()?;
    //#[cfg(feature = "bfgs")]
    //m.add_class::<PyBfgsJacSolver>()?;
    //#[cfg(feature = "ceres")]
    //m.add_class::<PyCeresJacSolver>()?;
    Ok(())
}
