use num_complex::Complex64;

use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray3};

use pyo3::prelude::*;

use better_panic::install;

use crate::python::circuit::PyCircuit;
use crate::utils::{
    matrix_distance_squared, matrix_distance_squared_jac, matrix_residuals, matrix_residuals_jac,
};

use crate::python::minimizers::*;

mod minimizers;

use crate::python::instantiators::*;

mod instantiators;

mod circuit;

use crate::permutation_matrix::{calc_permutation_matrix, swap_bit};

mod gate;

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

    #[pyfn(m, "calc_permutation_matrix")]
    fn calc_permutation_matrix_py(
        py: Python,
        num_qubits: usize,
        location: Vec<usize>,
    ) -> Py<PyArray2<Complex64>> {
        PyArray2::from_array(py, &calc_permutation_matrix(num_qubits, location)).to_owned()
    }

    #[pyfn(m, "swap_bit")]
    fn swap_bit_py(i: usize, j: usize, b: usize) -> usize {
        swap_bit(i, j, b)
    }

    #[pyfn(m, "matrix_distance_squared")]
    fn matrix_distance_squared_py(
        a: PyReadonlyArray2<Complex64>,
        b: PyReadonlyArray2<Complex64>,
    ) -> f64 {
        matrix_distance_squared(a.as_array(), b.as_array())
    }
    #[pyfn(m, "matrix_distance_squared_jac")]
    fn matrix_distance_squared_jac_py(
        a: PyReadonlyArray2<Complex64>,
        b: PyReadonlyArray2<Complex64>,
        jacs: PyReadonlyArray3<Complex64>,
    ) -> (f64, Vec<f64>) {
        matrix_distance_squared_jac(a.as_array(), b.as_array(), jacs.as_array())
    }
    #[pyfn(m, "matrix_residuals")]
    fn matrix_residuals_py(
        a: PyReadonlyArray2<Complex64>,
        b: PyReadonlyArray2<Complex64>,
        eye: PyReadonlyArray2<f64>,
    ) -> Vec<f64> {
        matrix_residuals(
            &a.to_owned_array(),
            &b.to_owned_array(),
            &eye.to_owned_array(),
        )
    }
    #[pyfn(m, "matrix_residuals_jac")]
    fn matrix_residuals_jac_py(
        py: Python,
        u: PyReadonlyArray2<Complex64>,
        m: PyReadonlyArray2<Complex64>,
        jacs: PyReadonlyArray3<Complex64>,
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
