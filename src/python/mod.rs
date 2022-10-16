use ndarray_linalg::c64;

use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray3};

use pyo3::prelude::*;

use crate::python::circuit::PyCircuit;
use crate::utils::{
    matrix_distance_squared, matrix_distance_squared_jac, matrix_residuals, matrix_residuals_jac,
};

use crate::python::minimizers::*;

mod minimizers;

use crate::python::instantiators::*;

mod instantiators;

mod circuit;

mod gate;

#[pymodule]
#[pyo3(name = "bqskitrs")]
fn bqskitrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyHilberSchmidtCostFn>()?;
    m.add_class::<PyHilberSchmidtResidualFn>()?;
    m.add_class::<PyBfgsJacSolver>()?;
    m.add_class::<PyCeresJacSolver>()?;
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyQFactorInstantiator>()?;

    #[pyfn(m)]
    #[pyo3(name = "matrix_distance_squared")]
    fn matrix_distance_squared_py(
        a: PyReadonlyArray2<c64>,
        b: PyReadonlyArray2<c64>,
    ) -> f64 {
        matrix_distance_squared(a.as_array(), b.as_array())
    }
    #[pyfn(m)]
    #[pyo3(name = "matrix_distance_squared_jac")]
    fn matrix_distance_squared_jac_py(
        a: PyReadonlyArray2<c64>,
        b: PyReadonlyArray2<c64>,
        jacs: PyReadonlyArray3<c64>,
    ) -> (f64, Vec<f64>) {
        matrix_distance_squared_jac(a.as_array(), b.as_array(), jacs.as_array())
    }
    #[pyfn(m)]
    #[pyo3(name = "matrix_residuals")]
    fn matrix_residuals_py(
        a: PyReadonlyArray2<c64>,
        b: PyReadonlyArray2<c64>,
        eye: PyReadonlyArray2<f64>,
    ) -> Vec<f64> {
        matrix_residuals(
            &a.to_owned_array(),
            &b.to_owned_array(),
            &eye.to_owned_array(),
        )
    }
    #[pyfn(m)]
    #[pyo3(name = "matrix_residuals_jac")]
    fn matrix_residuals_jac_py(
        py: Python,
        u: PyReadonlyArray2<c64>,
        m: PyReadonlyArray2<c64>,
        jacs: PyReadonlyArray3<c64>,
    ) -> Py<PyArray2<f64>> {
        PyArray2::from_array(
            py,
            &matrix_residuals_jac(
                &u.to_owned_array(),
                &m.to_owned_array(),
                &jacs.to_owned_array(),
            ),
        )
        .to_owned()
    }
    Ok(())
}
