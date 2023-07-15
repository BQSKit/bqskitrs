use ndarray::{Array2, Array1};
use ndarray_linalg::c64;
use ndarray_einsum_beta::einsum;
use crate::squaremat::*;

use crate::{
    ir::circuit::Circuit,
    ir::gates::{Gradient, Unitary},
    utils::{matrix_distance_squared, matrix_residuals, matrix_residuals_jac, state_infidelity, state_residuals, state_residuals_jac},
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
        self.as_ref().get_residuals(params)
    }

    fn num_residuals(&self) -> usize {
        self.as_ref().num_residuals()
    }
}

pub trait DifferentiableResidualFn: ResidualFn {
    fn get_grad(&self, params: &[f64]) -> Array2<f64>;
    fn get_residuals_and_grad(&self, params: &[f64]) -> (Vec<f64>, Array2<f64>) {
        (self.get_residuals(params), self.get_grad(params))
    }
}

impl<T> DifferentiableResidualFn for Box<T>
where
    T: DifferentiableResidualFn + Sized,
{
    fn get_grad(&self, params: &[f64]) -> Array2<f64> {
        self.as_ref().get_grad(params)
    }

    fn get_residuals_and_grad(&self, params: &[f64]) -> (Vec<f64>, Array2<f64>) {
        self.as_ref().get_residuals_and_grad(params)
    }
}

#[derive(Clone)]
pub struct HilbertSchmidtResidualFn {
    circ: Circuit,
    target: Array2<c64>,
    eye: Array2<f64>,
}

impl HilbertSchmidtResidualFn {
    pub fn new(circ: Circuit, target: Array2<c64>) -> Self {
        let size = target.shape()[0];
        HilbertSchmidtResidualFn {
            circ,
            target,
            eye: Array2::eye(size),
        }
    }

    pub fn is_sendable(&self) -> bool {
        self.circ.is_sendable()
    }
}

impl CostFn for HilbertSchmidtResidualFn {
    fn get_cost(&self, params: &[f64]) -> f64 {
        let calculated = self.circ.get_utry(params, &self.circ.constant_gates);
        matrix_distance_squared(self.target.view(), calculated.view())
    }
}

impl ResidualFn for HilbertSchmidtResidualFn {
    fn get_residuals(&self, params: &[f64]) -> Vec<f64> {
        let m = self.circ.get_utry(params, &self.circ.constant_gates);
        matrix_residuals(&self.target, &m, &self.eye)
    }

    fn num_residuals(&self) -> usize {
        let size = self.target.len();
        size * 2
    }
}

impl DifferentiableResidualFn for HilbertSchmidtResidualFn {
    fn get_grad(&self, params: &[f64]) -> Array2<f64> {
        let (m, j) = self
            .circ
            .get_utry_and_grad(params, &self.circ.constant_gates);
        matrix_residuals_jac(&self.target, &m, &j)
    }

    fn get_residuals_and_grad(&self, params: &[f64]) -> (Vec<f64>, Array2<f64>) {
        let (m, j) = self
            .circ
            .get_utry_and_grad(params, &self.circ.constant_gates);
        (
            matrix_residuals(&self.target, &m, &self.eye),
            matrix_residuals_jac(&self.target, &m, &j),
        )
    }
}

#[derive(Clone)]
pub struct HilbertSchmidtStateResidualFn {
    circ: Circuit,
    target: Array1<c64>,
    eye: Array2<f64>,
}

impl HilbertSchmidtStateResidualFn {
    pub fn new(circ: Circuit, target: Array1<c64>) -> Self {
        let size = target.shape()[0];
        HilbertSchmidtStateResidualFn {
            circ,
            target,
            eye: Array2::eye(size),
        }
    }

    pub fn is_sendable(&self) -> bool {
        self.circ.is_sendable()
    }
}

impl CostFn for HilbertSchmidtStateResidualFn {
    fn get_cost(&self, params: &[f64]) -> f64 {
        let calculated = self.circ.get_state(params, &self.circ.constant_gates);
        state_infidelity(self.target.view(), calculated.view())
    }
}

impl ResidualFn for HilbertSchmidtStateResidualFn {
    fn get_residuals(&self, params: &[f64]) -> Vec<f64> {
        let m = self.circ.get_state(params, &self.circ.constant_gates);
        state_residuals(self.target.view(), m.view())
    }

    fn num_residuals(&self) -> usize {
        self.target.len()
    }
}

impl DifferentiableResidualFn for HilbertSchmidtStateResidualFn {
    fn get_grad(&self, params: &[f64]) -> Array2<f64> {
        let (m, j) = self
            .circ
            .get_state_and_grads(params, &self.circ.constant_gates);
        state_residuals_jac(self.target.view(), m.view(), j.view())
    }

    fn get_residuals_and_grad(&self, params: &[f64]) -> (Vec<f64>, Array2<f64>) {
        let (m, j) = self
            .circ
            .get_state_and_grads(params, &self.circ.constant_gates);
        (
            state_residuals(self.target.view(), m.view()),
            state_residuals_jac(self.target.view(), m.view(), j.view())
        )
    }
}

#[derive(Clone)]
pub struct HilbertSchmidtSystemResidualFn {
    circ: Circuit,
    target: Array2<c64>,
    eye: Array2<f64>,
    vec_count: u32,
}

impl HilbertSchmidtSystemResidualFn {
    pub fn new(circ: Circuit, target: Array2<c64>, vec_count: u32) -> Self {
        let size = target.shape()[0];
        HilbertSchmidtSystemResidualFn {
            circ,
            target,
            eye: Array2::eye(size),
            vec_count: vec_count,
        }
    }

    pub fn is_sendable(&self) -> bool {
        self.circ.is_sendable()
    }
}

impl CostFn for HilbertSchmidtSystemResidualFn {
    fn get_cost(&self, params: &[f64]) -> f64 {
        let calculated = self.circ.get_utry(params, &self.circ.constant_gates);
        let a = self.target.view();
        let b = calculated.view();
        let prod = einsum("ij,ij->", &[&a, &b.conj()]).unwrap();
        let norm = prod.sum().norm() / self.vec_count as f64;
        1f64 - norm
    }
}

impl ResidualFn for HilbertSchmidtSystemResidualFn {
    fn get_residuals(&self, params: &[f64]) -> Vec<f64> {
        let m = self.circ.get_utry(params, &self.circ.constant_gates);
        matrix_residuals(&self.target, &m, &self.eye)
    }

    fn num_residuals(&self) -> usize {
        let size = self.target.len();
        size * 2
    }
}

impl DifferentiableResidualFn for HilbertSchmidtSystemResidualFn {
    fn get_grad(&self, params: &[f64]) -> Array2<f64> {
        let (m, j) = self
            .circ
            .get_utry_and_grad(params, &self.circ.constant_gates);
        matrix_residuals_jac(&self.target, &m, &j)
    }

    fn get_residuals_and_grad(&self, params: &[f64]) -> (Vec<f64>, Array2<f64>) {
        let (m, j) = self
            .circ
            .get_utry_and_grad(params, &self.circ.constant_gates);
        (
            matrix_residuals(&self.target, &m, &self.eye),
            matrix_residuals_jac(&self.target, &m, &j),
        )
    }
}

pub enum ResidualFunction {
    HilbertSchmidtSystem(Box<HilbertSchmidtSystemResidualFn>),
    HilbertSchmidtState(Box<HilbertSchmidtStateResidualFn>),
    HilbertSchmidt(Box<HilbertSchmidtResidualFn>),
    Dynamic(Box<dyn DifferentiableResidualFn>),
}

impl ResidualFunction {
    pub fn is_sendable(&self) -> bool {
        match self {
            Self::HilbertSchmidtSystem(hs) => hs.is_sendable(),
            Self::HilbertSchmidtState(hs) => hs.is_sendable(),
            Self::HilbertSchmidt(hs) => hs.is_sendable(),
            Self::Dynamic(_) => false,
        }
    }
}

impl CostFn for ResidualFunction {
    fn get_cost(&self, params: &[f64]) -> f64 {
        match self {
            Self::HilbertSchmidtSystem(hs) => hs.get_cost(params),
            Self::HilbertSchmidtState(hs) => hs.get_cost(params),
            Self::HilbertSchmidt(hs) => hs.get_cost(params),
            Self::Dynamic(d) => d.get_cost(params),
        }
    }
}

impl ResidualFn for ResidualFunction {
    fn get_residuals(&self, params: &[f64]) -> Vec<f64> {
        match self {
            Self::HilbertSchmidtSystem(hs) => hs.get_residuals(params),
            Self::HilbertSchmidtState(hs) => hs.get_residuals(params),
            Self::HilbertSchmidt(hs) => hs.get_residuals(params),
            Self::Dynamic(d) => d.get_residuals(params),
        }
    }

    fn num_residuals(&self) -> usize {
        match self {
            Self::HilbertSchmidtSystem(hs) => hs.num_residuals(),
            Self::HilbertSchmidtState(hs) => hs.num_residuals(),
            Self::HilbertSchmidt(hs) => hs.num_residuals(),
            Self::Dynamic(d) => d.num_residuals(),
        }
    }
}

impl DifferentiableResidualFn for ResidualFunction {
    fn get_grad(&self, params: &[f64]) -> Array2<f64> {
        match self {
            Self::HilbertSchmidtSystem(hs) => hs.get_grad(params),
            Self::HilbertSchmidtState(hs) => hs.get_grad(params),
            Self::HilbertSchmidt(hs) => hs.get_grad(params),
            Self::Dynamic(d) => d.get_grad(params),
        }
    }

    fn get_residuals_and_grad(&self, params: &[f64]) -> (Vec<f64>, Array2<f64>) {
        match self {
            Self::HilbertSchmidtSystem(hs) => hs.get_residuals_and_grad(params),
            Self::HilbertSchmidtState(hs) => hs.get_residuals_and_grad(params),
            Self::HilbertSchmidt(hs) => hs.get_residuals_and_grad(params),
            Self::Dynamic(d) => d.get_residuals_and_grad(params),
        }
    }
}
