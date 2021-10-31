use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use bqskitrs::circuit::Circuit;
use bqskitrs::operation::Operation;
use bqskitrs::gates::*;
use bqskitrs::instantiators::*;
use bqskitrs::{r, i};

use ndarray::Array2;
use num_complex::Complex64;

use std::f64::consts::{E, PI};

pub fn qft(n: usize) -> Array2<Complex64> {
    let root = r!(E).powc(i!(2f64) * PI / n as f64);
    Array2::from_shape_fn((n, n), |(x, y)| root.powf((x * y) as f64)) / (n as f64).sqrt()
}

extern "C" {
    fn srand(seed: u32);
}

fn optimize_qft4(instantiator: &QFactorInstantiator, circ: Circuit, target: Array2<Complex64>, x0: &[f64]) {
    let _x = instantiator.instantiate(circ, target, x0);
}

fn make_qft4_problem() -> (QFactorInstantiator, Circuit, Array2<Complex64>, Vec<f64>) {
    let mut ops = vec![];
    let mut constant_gates = vec![];
    // CNOT
    let v: Vec<Complex64> = vec![1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,0.,1., 0.,0.,1.,0.].iter().map(|i| Complex64::new(*i, 0.0)).collect();
    constant_gates.push(Array2::from_shape_vec((4,4), v).unwrap());
    let positions = vec![2,0,1,2,2,0,1,0,2,1,0,1,2,1,2,0];
    // Fill in the first layer
    for i in 0..4 {
        ops.push(Operation::new(VariableUnitaryGate::new(1, vec![2]).into(), vec![i], vec![0.0; 8]));
    }
    for position in positions {
        ops.push(Operation::new(ConstantGate::new(0, 2).into(), vec![position, position + 1], vec![]));
        ops.push(Operation::new(RXGate::new().into(), vec![position], vec![0.0]));
        ops.push(Operation::new(RZGate::new().into(), vec![position], vec![0.0]));
        ops.push(Operation::new(RXGate::new().into(), vec![position], vec![0.0]));
        ops.push(Operation::new(RZGate::new().into(), vec![position], vec![0.0]));
        ops.push(Operation::new(VariableUnitaryGate::new(1, vec![2]).into(), vec![position + 1], vec![0.0; 8]));
    }
    let circ = Circuit::new(4, vec![2; 4], ops, constant_gates);
    let instantiator = QFactorInstantiator::new(None, None, None, None, None, None, None);

    let x0 = vec![0.; circ.num_params()];
    (instantiator, circ, qft(16), x0)
}

fn bench(c: &mut Criterion) {
    // Set random seed for reproducability
    unsafe { srand(21211411) }
    

    // Setup (construct data, allocate memory, etc)
    let (instantiator, circ, target, x0) = make_qft4_problem();
    let mut group = c.benchmark_group("qfactor");
   
    group
    .sample_size(20)
    .bench_function(
        "optimize_qft4",
        |b| b.iter(|| {
            optimize_qft4(&instantiator, circ.clone(), target.clone(), &x0);
        }),
    );
}

criterion_group!(benches, bench);
criterion_main!(benches);