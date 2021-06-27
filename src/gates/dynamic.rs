use std::fmt;

use ndarray::ArrayViewMut2;
use num_complex::Complex64;
use squaremat::SquareMatrix;

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

    fn get_utry(&self, params: &[f64], const_gates: &[SquareMatrix]) -> SquareMatrix {
        (**self).get_utry(params, const_gates)
    }
}

impl<T> Gradient for Box<T>
where
    T: DynGate,
{
    fn get_grad(&self, params: &[f64], const_gates: &[SquareMatrix]) -> Vec<SquareMatrix> {
        (**self).get_grad(params, const_gates)
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        const_gates: &[SquareMatrix],
    ) -> (SquareMatrix, Vec<SquareMatrix>) {
        (**self).get_utry_and_grad(params, const_gates)
    }
}

impl<T> Size for Box<T>
where
    T: DynGate,
{
    fn get_size(&self) -> usize {
        (**self).get_size()
    }
}

impl<T> Optimize for Box<T>
where
    T: Optimize,
{
    fn optimize(&self, env_matrix: ArrayViewMut2<Complex64>) -> Vec<f64> {
        (**self).optimize(env_matrix)
    }
}
