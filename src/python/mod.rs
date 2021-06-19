use num_complex::Complex64;

use numpy::PyArray2;

use pyo3::prelude::*;

use better_panic::install;

use squaremat::SquareMatrix;

use crate::python::circuit::PyCircuit;
use crate::utils::{
    matrix_distance_squared, matrix_distance_squared_jac, matrix_residuals, matrix_residuals_jac,
};

use crate::python::minimizers::*;

mod minimizers;

use crate::python::instantiators::*;

mod instantiators;

mod circuit;

pub type PySquareMatrix = PyArray2<Complex64>;

#[pymodule]
fn bqskitrs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Install better panic for better tracebacks
    install();

    m.add_class::<PyHilberSchmidtCostFn>()?;
    m.add_class::<PyHilberSchmidtResidualFn>()?;
    m.add_class::<PyBfgsJacSolver>()?;
    m.add_class::<PyCeresJacSolver>()?;
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyQFactorInstantiator>()?;

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
    Ok(())
}
