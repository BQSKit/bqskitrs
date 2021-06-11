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

#[cfg(any(feature = "ceres", feature = "bfgs"))]
mod minimizers;

pub type PySquareMatrix = PyArray2<Complex64>;

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
    Ok(())
}
