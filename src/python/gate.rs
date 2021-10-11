use ndarray::{Array2, Array3};
use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;

use std::fmt;

use crate::gates::{DynGate, Gradient, Optimize, Size, Unitary};

pub struct PyGate {
    gate: PyObject,
}

impl PyGate {
    pub fn new(gate: PyObject) -> Self {
        PyGate { gate }
    }
}

impl fmt::Debug for PyGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let gil = Python::acquire_gil();
        let py = gil.python();
        f.write_str(self.gate.as_ref(py).repr().unwrap().to_str().unwrap())
    }
}

impl DynGate for PyGate {}

impl Unitary for PyGate {
    fn num_params(&self) -> usize {
        let gil = Python::acquire_gil();
        let py = gil.python();
        self.gate
            .getattr(py, "num_params")
            .expect("Failed to call num_params on passed gate.")
            .extract::<usize>(py)
            .expect("Return of num_params could not be converted into integral type.")
    }

    fn get_utry(&self, params: &[f64], _const_gates: &[Array2<Complex64>]) -> Array2<Complex64> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let args = (PyArray1::from_slice(py, params).to_object(py),);
        let pyutry = self
            .gate
            .call_method1(py, "get_unitary", args)
            .expect("Failed to call get_unitary on passed gate.");
        let pyarray = match pyutry.as_ref(py).hasattr("numpy").unwrap() {
            true => pyutry
                .getattr(py, "numpy")
                .expect("Failed to convert UnitaryMatrix to ndarray."),
            false => pyutry,
        }
        .extract::<Py<PyArray2<Complex64>>>(py)
        .expect("Failed to convert return of get array into complex matrix.");
        pyarray.as_ref(py).to_owned_array()
    }
}

impl Gradient for PyGate {
    fn get_grad(&self, params: &[f64], _const_gates: &[Array2<Complex64>]) -> Array3<Complex64> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let args = (PyArray1::from_slice(py, params).to_object(py),);
        let pygrads = self
            .gate
            .call_method1(py, "get_grad", args)
            .expect("Failed to call get_grad on passed gate.")
            .extract::<PyObject>(py)
            .unwrap();
        match pygrads.as_ref(py).hasattr("numpy").unwrap() {
            true => pygrads
                .getattr(py, "numpy")
                .expect("Failed to convert UnitaryMatrix to ndarray."),
            false => pygrads,
        }
        .extract::<Py<PyArray3<Complex64>>>(py)
        .expect("Failed to convert return of get_grad into complex matrix.")
        .as_ref(py)
        .to_owned_array()
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        _const_gates: &[Array2<Complex64>],
    ) -> (Array2<Complex64>, Array3<Complex64>) {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let args = (PyArray1::from_slice(py, params).to_object(py),);
        let (pyutry, pygrads) = self
            .gate
            .call_method1(py, "get_unitary_and_grad", args)
            .expect("Failed to call get_unitary_and_grad on passed gate.")
            .extract::<(PyObject, PyObject)>(py)
            .expect("Failed to convert return of get_unitary_and_grad.");

        let pyarray = match pyutry.as_ref(py).hasattr("numpy").unwrap() {
            true => pyutry
                .getattr(py, "numpy")
                .expect("Failed to convert UnitaryMatrix to ndarray."),
            false => pyutry,
        }
        .extract::<Py<PyArray2<Complex64>>>(py)
        .expect("Failed to convert return of get array into complex matrix.");

        let grads = match pygrads.as_ref(py).hasattr("numpy").unwrap() {
            true => pygrads
                .getattr(py, "numpy")
                .expect("Failed to convert UnitaryMatrix to ndarray."),
            false => pygrads,
        }
        .extract::<Py<PyArray3<Complex64>>>(py)
        .expect("Failed to convert return of get_grad into complex matrix.");

        (
            pyarray.as_ref(py).to_owned_array(),
            grads.as_ref(py).to_owned_array(),
        )
    }
}

impl Size for PyGate {
    fn num_qudits(&self) -> usize {
        let gil = Python::acquire_gil();
        let py = gil.python();
        self.gate
            .call_method0(py, "num_qudits")
            .expect("Failed to call num_qudits on passed gate.")
            .extract::<usize>(py)
            .expect("Failed to convert the return of num_qudits to an integer.")
    }
}

impl Optimize for PyGate {
    fn optimize(&self, env_matrix: ndarray::ArrayViewMut2<Complex64>) -> Vec<f64> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let args = (PyArray2::from_array(py, &env_matrix.to_owned()).to_object(py),);
        self.gate
            .call_method1(py, "optimize", args)
            .expect("Failed to call optimize on passed gate.")
            .extract::<Vec<f64>>(py)
            .expect("Failed to convert the return of optimize to a list of floats.")
    }
}
