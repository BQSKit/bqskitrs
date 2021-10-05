use std::fmt;

use ndarray::{Array2, Array3, ArrayViewMut2};
use num_complex::Complex64;

use super::{Gradient, Optimize, Size, Unitary};

pub trait DynGate: Unitary + Gradient + Size + Optimize + fmt::Debug {}

impl<T> DynGate for Box<T> where T: DynGate {}

impl<T> Unitary for Box<T>
where
    T: DynGate,
{
    fn num_params(&self) -> usize {
        (**self).num_params()
    }

    fn get_utry(&self, params: &[f64], const_gates: &[Array2<Complex64>]) -> Array2<Complex64> {
        (**self).get_utry(params, const_gates)
    }
}

impl<T> Gradient for Box<T>
where
    T: DynGate,
{
    fn get_grad(&self, params: &[f64], const_gates: &[Array2<Complex64>]) -> Array3<Complex64> {
        (**self).get_grad(params, const_gates)
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        const_gates: &[Array2<Complex64>],
    ) -> (Array2<Complex64>, Array3<Complex64>) {
        (**self).get_utry_and_grad(params, const_gates)
    }
}

impl<T> Size for Box<T>
where
    T: DynGate,
{
    fn num_qudits(&self) -> usize {
        (**self).num_qudits()
    }
}

impl<T> Optimize for Box<T>
where
    T: Optimize + DynGate,
{
    fn optimize(&self, env_matrix: ArrayViewMut2<Complex64>) -> Vec<f64> {
        (**self).optimize(env_matrix)
    }
}
