use crate::{
    ir::circuit::Circuit,
    ir::gates::{Gradient, Unitary},
    utils::{matrix_distance_squared, matrix_distance_squared_jac, state_infidelity, state_infidelity_jac, matrix_distance_system_squared, matrix_distance_system_squared_jac},
};

use enum_dispatch::enum_dispatch;
use ndarray::{Array2, Array1};
use ndarray_linalg::c64;

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
    target: Array2<c64>,
}

impl HilbertSchmidtCostFn {
    pub fn new(circ: Circuit, target: Array2<c64>) -> Self {
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

#[derive(Clone)]
pub struct HilbertSchmidtStateCostFn {
    circ: Circuit,
    target: Array1<c64>,
}

impl HilbertSchmidtStateCostFn {
    pub fn new(circ: Circuit, target: Array1<c64>) -> Self {
        HilbertSchmidtStateCostFn { circ, target }
    }

    pub fn is_sendable(&self) -> bool {
        self.circ.is_sendable()
    }
}

impl CostFn for HilbertSchmidtStateCostFn {
    fn get_cost(&self, params: &[f64]) -> f64 {
        let calculated = self.circ.get_state(params, &self.circ.constant_gates);
        state_infidelity(self.target.view(), calculated.view())
    }
}

impl DifferentiableCostFn for HilbertSchmidtStateCostFn {
    fn get_grad(&self, params: &[f64]) -> Vec<f64> {
        let (m, j) = self
            .circ
            .get_state_and_grads(params, &self.circ.constant_gates);
        state_infidelity_jac(self.target.view(), m.view(), j.view()).1
    }

    fn get_cost_and_grad(&self, params: &[f64]) -> (f64, Vec<f64>) {
        let (m, j) = self
            .circ
            .get_state_and_grads(params, &self.circ.constant_gates);
        state_infidelity_jac(self.target.view(), m.view(), j.view())
    }
}

#[derive(Clone)]
pub struct HilbertSchmidtSystemCostFn {
    circ: Circuit,
    target: Array2<c64>,
    vec_count: u32,
}

impl HilbertSchmidtSystemCostFn {
    pub fn new(circ: Circuit, target: Array2<c64>, vec_count: u32) -> Self {
        HilbertSchmidtSystemCostFn { circ, target, vec_count }
    }

    pub fn is_sendable(&self) -> bool {
        self.circ.is_sendable()
    }
}

impl CostFn for HilbertSchmidtSystemCostFn {
    fn get_cost(&self, params: &[f64]) -> f64 {
        let calculated = self.circ.get_utry(params, &self.circ.constant_gates);
        matrix_distance_system_squared(self.target.view(), calculated.view(), self.vec_count)
    }
}

impl DifferentiableCostFn for HilbertSchmidtSystemCostFn {
    fn get_grad(&self, params: &[f64]) -> Vec<f64> {
        let (m, j) = self
            .circ
            .get_utry_and_grad(params, &self.circ.constant_gates);
        matrix_distance_system_squared_jac(self.target.view(), m.view(), j.view(), self.vec_count).1
    }

    fn get_cost_and_grad(&self, params: &[f64]) -> (f64, Vec<f64>) {
        let (m, j) = self
            .circ
            .get_utry_and_grad(params, &self.circ.constant_gates);
        matrix_distance_system_squared_jac(self.target.view(), m.view(), j.view(), self.vec_count)
    }
}

pub enum CostFunction {
    HilbertSchmidt(HilbertSchmidtCostFn),
    HilbertSchmidtState(HilbertSchmidtStateCostFn),
    HilbertSchmidtSystem(HilbertSchmidtSystemCostFn),
    Dynamic(Box<dyn DifferentiableCostFn>),
}

impl CostFunction {
    pub fn is_sendable(&self) -> bool {
        match self {
            CostFunction::HilbertSchmidt(hs) => hs.is_sendable(),
            CostFunction::HilbertSchmidtState(hs) => hs.is_sendable(),
            CostFunction::HilbertSchmidtSystem(hs) => hs.is_sendable(),
            CostFunction::Dynamic(_) => false,
        }
    }
}

impl CostFn for CostFunction {
    fn get_cost(&self, params: &[f64]) -> f64 {
        match self {
            Self::HilbertSchmidt(hs) => hs.get_cost(params),
            Self::HilbertSchmidtState(hs) => hs.get_cost(params),
            Self::HilbertSchmidtSystem(hs) => hs.get_cost(params),
            Self::Dynamic(d) => d.get_cost(params),
        }
    }
}

impl DifferentiableCostFn for CostFunction {
    fn get_grad(&self, params: &[f64]) -> Vec<f64> {
        match self {
            Self::HilbertSchmidt(hs) => hs.get_grad(params),
            Self::HilbertSchmidtState(hs) => hs.get_grad(params),
            Self::HilbertSchmidtSystem(hs) => hs.get_grad(params),
            Self::Dynamic(d) => d.get_grad(params),
        }
    }

    fn get_cost_and_grad(&self, params: &[f64]) -> (f64, Vec<f64>) {
        match self {
            Self::HilbertSchmidt(hs) => hs.get_cost_and_grad(params),
            Self::HilbertSchmidtState(hs) => hs.get_cost_and_grad(params),
            Self::HilbertSchmidtSystem(hs) => hs.get_cost_and_grad(params),
            Self::Dynamic(d) => d.get_cost_and_grad(params),
        }
    }
}
