use crate::{
    circuit::Circuit,
    gates::{Gradient, Unitary},
    utils::{matrix_distance_squared, matrix_distance_squared_jac},
};

use enum_dispatch::enum_dispatch;
use ndarray::Array2;
use num_complex::Complex64;

/// Trait defining the signature of a cost function used by minimizers.
#[enum_dispatch]
pub trait CostFn: Send {
    fn get_cost(&self, params: &[f64]) -> f64;
}

impl<T> CostFn for Box<T>
where
    T: CostFn,
{
    fn get_cost(&self, params: &[f64]) -> f64 {
        self.as_ref().get_cost(params)
    }
}

pub trait DifferentiableCostFn: CostFn {
    fn get_grad(&self, params: &[f64]) -> Vec<f64>;
    fn get_cost_and_grad(&self, params: &[f64]) -> (f64, Vec<f64>) {
        (self.get_cost(params), self.get_grad(params))
    }
}

#[derive(Clone)]
pub struct HilbertSchmidtCostFn {
    circ: Circuit,
    target: Array2<Complex64>,
}

impl HilbertSchmidtCostFn {
    pub fn new(circ: Circuit, target: Array2<Complex64>) -> Self {
        HilbertSchmidtCostFn { circ, target }
    }

    pub fn is_sendable(&self) -> bool {
        self.circ.is_sendable()
    }
}

impl CostFn for HilbertSchmidtCostFn {
    fn get_cost(&self, params: &[f64]) -> f64 {
        let calculated = self.circ.get_utry(params, &self.circ.constant_gates);
        matrix_distance_squared(self.target.view(), calculated.view())
    }
}

impl DifferentiableCostFn for HilbertSchmidtCostFn {
    fn get_grad(&self, params: &[f64]) -> Vec<f64> {
        let (m, j) = self
            .circ
            .get_utry_and_grad(params, &self.circ.constant_gates);
        matrix_distance_squared_jac(self.target.view(), m.view(), j.view()).1
    }

    fn get_cost_and_grad(&self, params: &[f64]) -> (f64, Vec<f64>) {
        let (m, j) = self
            .circ
            .get_utry_and_grad(params, &self.circ.constant_gates);
        matrix_distance_squared_jac(self.target.view(), m.view(), j.view())
    }
}

pub enum CostFunction {
    HilbertSchmidt(HilbertSchmidtCostFn),
    Dynamic(Box<dyn DifferentiableCostFn>),
}

impl CostFunction {
    pub fn is_sendable(&self) -> bool {
        match self {
            CostFunction::HilbertSchmidt(hs) => hs.is_sendable(),
            CostFunction::Dynamic(_) => false,
        }
    }
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
