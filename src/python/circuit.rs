use crate::circuit::Circuit;
use crate::gates::Gradient;
use crate::gates::Unitary;
use crate::gates::*;
use crate::operation::Operation;

use ndarray::Array3;
use num_complex::Complex64;

use numpy::PyArray2;
use numpy::PyArray3;
use pyo3::exceptions;
use pyo3::{prelude::*, types::PyIterator};
use squaremat::SquareMatrix;

fn pygate_to_native(pygate: &PyAny, constant_gates: &mut Vec<SquareMatrix>) -> PyResult<Gate> {
    let cls = pygate.getattr("__class__")?;
    let dunder_name = cls.getattr("__name__")?;
    let name = dunder_name.extract::<&str>()?;
    match name {
        "RXGate" => Ok(RXGate::new().into()),
        "RYGate" => Ok(RYGate::new().into()),
        "RZGate" => Ok(RZGate::new().into()),
        "U1Gate" => Ok(U1Gate::new().into()),
        "U2Gate" => Ok(U2Gate::new().into()),
        "U3Gate" => Ok(U3Gate::new().into()),
        "VariableUnitaryGate" => {
            let size = pygate.getattr("size")?.extract::<usize>()?;
            let radixes = pygate.getattr("radixes")?.extract::<Vec<usize>>()?;
            Ok(VariableUnitaryGate::new(size, radixes).into())
        }
        _ => {
            if pygate.getattr("num_params")?.extract::<usize>()? == 0 {
                let args: Vec<f64> = vec![];
                let pyobj = pygate.call_method("get_unitary", (args,), None)?;
                let pymat = pyobj
                    .call_method0("get_numpy")?
                    .extract::<&PyArray2<Complex64>>()?;
                let mat = pymat.to_owned_array();
                let gate_size = pygate.getattr("size")?.extract::<usize>()?;
                let index = constant_gates.len();
                constant_gates.push(SquareMatrix::from_ndarray(mat).T());
                Ok(ConstantGate::new(index, gate_size).into())
            } else {
                Err(exceptions::PyValueError::new_err(format!(
                    "Unknown gate {}",
                    name
                )))
            }
        }
    }
}

impl<'source> FromPyObject<'source> for Circuit {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let size = ob.call_method0("get_size")?.extract::<usize>()?;
        let radixes = ob.call_method0("get_radixes")?.extract::<Vec<usize>>()?;
        let circ_iter = ob.call_method0("operations")?;
        let iter = PyIterator::from_object(py, circ_iter)?;
        let mut ops = vec![];
        let mut constant_gates = vec![];
        for operation in iter {
            let op = operation?;
            let pygate = op.getattr("gate")?;
            let location = op.getattr("location")?.extract::<Vec<usize>>()?;
            let params = op.getattr("params")?.extract::<Vec<f64>>()?;
            let gate = pygate_to_native(pygate, &mut constant_gates)?;
            ops.push(Operation::new(gate, location, params));
        }
        Ok(Circuit::new(size, radixes, ops, constant_gates))
    }
}

#[pyclass(name = "Circuit", subclass, module = "bqskitrs")]
pub struct PyCircuit {
    circ: Circuit,
}

#[pymethods]
impl PyCircuit {
    #[new]
    pub fn new(circ: Circuit) -> Self {
        PyCircuit { circ }
    }

    pub fn get_unitary(&self, py: Python, params: Vec<f64>) -> Py<PyArray2<Complex64>> {
        PyArray2::from_array(
            py,
            &self
                .circ
                .get_utry(&params, &self.circ.constant_gates)
                .into_ndarray(),
        )
        .to_owned()
    }

    pub fn get_grad(&self, py: Python, params: Vec<f64>) -> Py<PyArray3<Complex64>> {
        let grad = self.circ.get_grad(&params, &self.circ.constant_gates);
        if grad.is_empty() {
            return PyArray3::zeros(py, (0, 0, 0), false).to_owned();
        }
        let size = grad[0].size;
        PyArray3::from_array(
            py,
            &Array3::from_shape_vec(
                (grad.len(), size, size),
                grad.into_iter().fold(vec![], |mut v, mat| {
                    v.extend(mat.into_vec());
                    v
                }),
            )
            .unwrap(),
        )
        .to_owned()
    }

    pub fn get_unitary_and_grad(
        &self,
        py: Python,
        params: Vec<f64>,
    ) -> (Py<PyArray2<Complex64>>, Py<PyArray3<Complex64>>) {
        let (utry, grad) = self
            .circ
            .get_utry_and_grad(&params, &self.circ.constant_gates);
        let size = utry.size;
        (
            PyArray2::from_array(py, &utry.into_ndarray()).to_owned(),
            PyArray3::from_array(
                py,
                &Array3::from_shape_vec(
                    (grad.len(), size, size),
                    grad.into_iter().fold(vec![], |mut v, mat| {
                        v.extend(mat.into_vec());
                        v
                    }),
                )
                .unwrap(),
            )
            .to_owned(),
        )
    }
}
