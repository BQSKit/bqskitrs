use squaremat::SquareMatrix;

use crate::{
    circuit::Circuit,
    gates::{Gradient, Unitary},
    utils::{matrix_distance_squared, matrix_distance_squared_jac},
};

use enum_dispatch::enum_dispatch;

/// Trait defining the signature of a cost function used by minimizers.
#[enum_dispatch]
pub trait CostFn {
    fn get_cost(&self, params: &[f64]) -> f64;
}

impl<T> CostFn for Box<T>
where
    T: CostFn,
{
    fn get_cost(&self, params: &[f64]) -> f64 {
        CostFn::get_cost(self, params)
    }
}

pub trait DifferentiableCostFn: CostFn {
    fn get_grad(&self, params: &[f64]) -> Vec<f64>;
    fn get_cost_and_grad(&self, params: &[f64]) -> (f64, Vec<f64>) {
        (self.get_cost(params), self.get_grad(params))
    }
}

pub struct HilbertSchmidtCostFn {
    circ: Circuit,
    target: SquareMatrix,
}

impl CostFn for HilbertSchmidtCostFn {
    fn get_cost(&self, params: &[f64]) -> f64 {
        let calculated = self.circ.get_utry(params, &self.circ.constant_gates);
        matrix_distance_squared(&self.target, &calculated)
    }
}

impl DifferentiableCostFn for HilbertSchmidtCostFn {
    fn get_grad(&self, params: &[f64]) -> Vec<f64> {
        let (m, j) = self
            .circ
            .get_utry_and_grad(params, &self.circ.constant_gates);
        matrix_distance_squared_jac(&self.target, &m, j).1
    }

    fn get_cost_and_grad(&self, params: &[f64]) -> (f64, Vec<f64>) {
        let (m, j) = self
            .circ
            .get_utry_and_grad(params, &self.circ.constant_gates);
        matrix_distance_squared_jac(&self.target, &m, j)
    }
}

pub enum CostFunction {
    HilbertSchmidt(HilbertSchmidtCostFn),
    Dynamic(Box<dyn DifferentiableCostFn>),
}

impl CostFn for CostFunction {
    fn get_cost(&self, params: &[f64]) -> f64 {
        match self {
            Self::HilbertSchmidt(hs) => hs.get_cost(params),
            Self::Dynamic(d) => d.get_cost(params),
        }
    }
}

impl DifferentiableCostFn for CostFunction {
    fn get_grad(&self, params: &[f64]) -> Vec<f64> {
        match self {
            Self::HilbertSchmidt(hs) => hs.get_grad(params),
            Self::Dynamic(d) => d.get_grad(params),
        }
    }

    fn get_cost_and_grad(&self, params: &[f64]) -> (f64, Vec<f64>) {
        match self {
            Self::HilbertSchmidt(hs) => hs.get_cost_and_grad(params),
            Self::Dynamic(d) => d.get_cost_and_grad(params),
        }
    }
}
