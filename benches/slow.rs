use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use bqskitrs::circuit::operation::Operation;
use bqskitrs::circuit::Circuit;
use bqskitrs::gates::*;
use bqskitrs::instantiators::*;
use bqskitrs::minimizers::*;

use ndarray::Array2;
use num_complex::Complex64;

use ndarray_npy::ReadNpyExt;

extern "C" {
    fn srand(seed: u32);
}

fn make_circuit(positions: &[(usize, usize)], num_qubits: usize, u3: bool) -> Circuit {
    let mut ops = vec![];
    let mut constant_gates = vec![];
    // CNOT
    let v: Vec<Complex64> = vec![
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
    ]
    .iter()
    .map(|i| Complex64::new(*i, 0.0))
    .collect();
    constant_gates.push(Array2::from_shape_vec((4, 4), v).unwrap());
    // Fill in the first row
    for qubit in 0..num_qubits {
        if u3 {
            ops.push((
                0,
                Operation::new(U3Gate::new().into(), vec![qubit], vec![0.0; 3]),
            ));
        } else {
            ops.push((
                0,
                Operation::new(
                    VariableUnitaryGate::new(1, vec![2]).into(),
                    vec![qubit],
                    vec![0.0; 8],
                ),
            ));
        }
    }

    for (cycle, position) in positions.iter().enumerate() {
        ops.push((
            cycle,
            Operation::new(
                ConstantGate::new(0, 2).into(),
                vec![position.0, position.1],
                vec![],
            ),
        ));
        ops.push((
            cycle,
            Operation::new(RXGate::new().into(), vec![position.0], vec![0.0]),
        ));
        ops.push((
            cycle,
            Operation::new(RZGate::new().into(), vec![position.0], vec![0.0]),
        ));
        ops.push((
            cycle,
            Operation::new(RXGate::new().into(), vec![position.0], vec![0.0]),
        ));
        ops.push((
            cycle,
            Operation::new(RZGate::new().into(), vec![position.0], vec![0.0]),
        ));
        if u3 {
            ops.push((
                cycle,
                Operation::new(U3Gate::new().into(), vec![position.1], vec![0.0; 3]),
            ));
        } else {
            ops.push((
                cycle,
                Operation::new(
                    VariableUnitaryGate::new(1, vec![2]).into(),
                    vec![position.1],
                    vec![0.0; 8],
                ),
            ));
        }
    }
    Circuit::new(
        num_qubits,
        vec![2; num_qubits],
        ops,
        constant_gates,
        bqskitrs::circuit::SimulationBackend::Matrix,
    )
}

fn optimize_ceres(minimizer: &CeresJacSolver, cost: &ResidualFunction, x0: &[f64]) -> Vec<f64> {
    minimizer.minimize(cost, x0)
}

fn optimize_qfactor(
    instantiator: &QFactorInstantiator,
    circ: &mut Circuit,
    target: Array2<Complex64>,
    x0: &[f64],
) -> Vec<f64> {
    instantiator.instantiate(circ, target, x0)
}

fn create_benches() -> Vec<(&'static str, Vec<(usize, usize)>, Vec<u8>, usize)> {
    // Set random seed for reproducability
    unsafe { srand(42) }

    // Positions of CNOTs in each circuit
    let qft4_partial = vec![(0, 3), (1, 2), (0,3), (0,3)];

    let qft4_bytes = include_bytes!("unitaries/qft4.npy");

    let benchs = vec![
        ("qft4_partial", qft4_partial, qft4_bytes.to_vec(), 4),
    ];
    return benchs;
}

fn bench_qfactor(c: &mut Criterion) {
    // Setup numerical optimizers
    let instantiator = QFactorInstantiator::new(None, None, None, None, None, None, None);

    let benchs = create_benches();
    // Run QFactor benchmarks
    let mut group = c.benchmark_group("qfactor");
    for (name, positions, npy, qubits) in benchs {
        let circ = make_circuit(&positions, qubits, false);
        let x0 = vec![0.0; circ.num_params()];
        let target = Array2::read_npy(&npy[..]).unwrap();
        let mut x = Vec::new();
        group.sample_size(10).bench_function(name, |b| {
            unsafe { srand(42) };
            b.iter(|| {
                x = optimize_qfactor(&instantiator, &mut circ.clone(), target.clone(), &x0);
            })
        });
        println!("{:?}", x);
    }
}

fn bench_ceres(c: &mut Criterion) {
    let benchs = create_benches();
    let minimizer = CeresJacSolver::new(1, 1e-6, 1e-10, false);
    // Run Ceres benchmarks
    let mut group = c.benchmark_group("ceres");
    for (name, positions, npy, qubits) in &benchs {
        let circ = make_circuit(positions, *qubits, true);
        let x0 = vec![0.0; circ.num_params()];
        let target = Array2::read_npy(&npy[..]).unwrap();
        let cost = HilbertSchmidtResidualFn::new(circ, target);
        let mut x = Vec::new();
        group.sample_size(10).bench_function(name.clone(), |b| {
            unsafe { srand(42) };
            b.iter(|| {
                x = optimize_ceres(
                    &minimizer,
                    &ResidualFunction::HilbertSchmidt(cost.clone()),
                    &x0,
                );
            })
        });
        println!("{:?}", x);
    }
}

criterion_group!(ceres, bench_ceres);
criterion_group!(qfactor, bench_qfactor);
criterion_main!(ceres, qfactor);
