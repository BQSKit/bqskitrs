use bqskitrs::circuit::Circuit;
use bqskitrs::operation::Operation;
use bqskitrs::gates::*;
use bqskitrs::minimizers::*;
use bqskitrs::{r, i};

use ndarray::Array2;
use num_complex::Complex64;

use std::f64::consts::{E, PI};

pub fn qft(n: usize) -> Array2<Complex64> {
    let root = r!(E).powc(i!(2f64) * PI / n as f64);
    Array2::from_shape_fn((n, n), |(x, y)| root.powf((x * y) as f64)) / (n as f64).sqrt()
}

fn main() {
    let mut ops = vec![];
    let mut constant_gates = vec![];
    // CNOT
    let v: Vec<Complex64> = vec![1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,0.,1., 0.,0.,1.,0.].iter().map(|i| Complex64::new(*i, 0.0)).collect();
    constant_gates.push(Array2::from_shape_vec((4,4), v).unwrap());
    let positions = vec![2,0,1,2,2,0,1,0,2,1,0,1,2,1,2,0];
    // Fill in the first layer
    for i in 0..4 {
        ops.push(Operation::new(U3Gate::new().into(), vec![i], vec![0.0; 3]));
    }
    for position in positions {
        ops.push(Operation::new(ConstantGate::new(0, 2).into(), vec![position, position + 1], vec![]));
        ops.push(Operation::new(RXGate::new().into(), vec![position], vec![0.0]));
        ops.push(Operation::new(RZGate::new().into(), vec![position], vec![0.0]));
        ops.push(Operation::new(RXGate::new().into(), vec![position], vec![0.0]));
        ops.push(Operation::new(RZGate::new().into(), vec![position], vec![0.0]));
        ops.push(Operation::new(U3Gate::new().into(), vec![position + 1], vec![0.0]));
    }
    let circ = Circuit::new(4, vec![2; 4], ops, constant_gates);
    let cost = HilbertSchmidtResidualFn::new(circ, qft(16));
    let minimizer = CeresJacSolver::new(1, 1e-6, 1e-10, false);
    let x0 = vec![0.0;124];
    let x = minimizer.minimize(ResidualFunction::HilbertSchmidt(cost), x0);
    assert!(x.iter().sum::<f64>() != 0.0)
}