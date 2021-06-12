use ndarray::Array2;
use squaremat::SquareMatrix;

use crate::{
    circuit::Circuit,
    gates::{Gradient, Unitary},
    utils::{matrix_distance_squared, matrix_residuals, matrix_residuals_jac},
};

use enum_dispatch::enum_dispatch;

use super::CostFn;

/// Trait defining the signature of a cost function used by minimizers.
#[enum_dispatch]
pub trait ResidualFn: CostFn {
    fn get_residuals(&self, params: &[f64]) -> Vec<f64>;
    fn num_residuals(&self) -> usize;
}

impl<T> ResidualFn for Box<T>
where
    T: ResidualFn + Sized,
{
    fn get_residuals(&self, params: &[f64]) -> Vec<f64> {
        ResidualFn::get_residuals(self, params)
    }

    fn num_residuals(&self) -> usize {
        ResidualFn::num_residuals(self)
    }
}

pub trait DifferentiableResidualFn: ResidualFn {
    fn get_grad(&self, params: &[f64]) -> Array2<f64>;
    fn get_residuals_and_grad(&self, params: &[f64]) -> (Vec<f64>, Array2<f64>) {
        (self.get_residuals(params), self.get_grad(params))
    }
}

#[derive(Clone)]
pub struct HilbertSchmidtResidualFn {
    circ: Circuit,
    target: SquareMatrix,
    eye: Array2<f64>,
}

impl CostFn for HilbertSchmidtResidualFn {
    fn get_cost(&self, params: &[f64]) -> f64 {
        let calculated = self.circ.get_utry(params, &self.circ.constant_gates);
        matrix_distance_squared(&self.target, &calculated)
    }
}

impl ResidualFn for HilbertSchmidtResidualFn {
    fn get_residuals(&self, params: &[f64]) -> Vec<f64> {
        let m = self.circ.get_utry(params, &self.circ.constant_gates);
        matrix_residuals(&self.target, &m, &self.eye)
    }

    fn num_residuals(&self) -> usize {
        self.target.size * self.target.size * 2
    }
}

impl DifferentiableResidualFn for HilbertSchmidtResidualFn {
    fn get_grad(&self, params: &[f64]) -> Array2<f64> {
        let (m, j) = self
            .circ
            .get_utry_and_grad(params, &self.circ.constant_gates);
        matrix_residuals_jac(&self.target, &m, &j)
    }
}

pub enum ResidualFunction {
    HilbertSchmidt(HilbertSchmidtResidualFn),
    Dynamic(Box<dyn DifferentiableResidualFn>),
}

impl CostFn for ResidualFunction {
    fn get_cost(&self, params: &[f64]) -> f64 {
        match self {
            Self::HilbertSchmidt(hs) => hs.get_cost(params),
            Self::Dynamic(d) => d.get_cost(params),
        }
    }
}

impl ResidualFn for ResidualFunction {
    fn get_residuals(&self, params: &[f64]) -> Vec<f64> {
        match self {
            Self::HilbertSchmidt(hs) => hs.get_residuals(params),
            Self::Dynamic(d) => d.get_residuals(params),
        }
    }

    fn num_residuals(&self) -> usize {
        match self {
            Self::HilbertSchmidt(hs) => hs.num_residuals(),
            Self::Dynamic(d) => d.num_residuals(),
        }
    }
}

impl DifferentiableResidualFn for ResidualFunction {
    fn get_grad(&self, params: &[f64]) -> Array2<f64> {
        match self {
            Self::HilbertSchmidt(hs) => hs.get_grad(params),
            Self::Dynamic(d) => d.get_grad(params),
        }
    }

    fn get_residuals_and_grad(&self, params: &[f64]) -> (Vec<f64>, Array2<f64>) {
        match self {
            Self::HilbertSchmidt(hs) => hs.get_residuals_and_grad(params),
            Self::Dynamic(d) => d.get_residuals_and_grad(params),
        }
    }
}
