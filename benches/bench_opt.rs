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

fn make_circuit(positions: &[usize], num_qubits: usize, u3: bool) -> Circuit {
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
                vec![*position, *position + 1],
                vec![],
            ),
        ));
        ops.push((
            cycle,
            Operation::new(RXGate::new().into(), vec![*position], vec![0.0]),
        ));
        ops.push((
            cycle,
            Operation::new(RZGate::new().into(), vec![*position], vec![0.0]),
        ));
        ops.push((
            cycle,
            Operation::new(RXGate::new().into(), vec![*position], vec![0.0]),
        ));
        ops.push((
            cycle,
            Operation::new(RZGate::new().into(), vec![*position], vec![0.0]),
        ));
        if u3 {
            ops.push((
                cycle,
                Operation::new(U3Gate::new().into(), vec![position + 1], vec![0.0; 3]),
            ));
        } else {
            ops.push((
                cycle,
                Operation::new(
                    VariableUnitaryGate::new(1, vec![2]).into(),
                    vec![position + 1],
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

fn optimize_ceres(minimizer: &CeresJacSolver, cost: &ResidualFunction, x0: &[f64]) {
    let _x = minimizer.minimize(cost, x0);
}

fn optimize_qfactor(
    instantiator: &QFactorInstantiator,
    circ: &mut Circuit,
    target: Array2<Complex64>,
    x0: &[f64],
) {
    let _x = instantiator.instantiate(circ, target, x0);
}

fn create_benches() -> Vec<(&'static str, Vec<usize>, Vec<u8>, usize)> {
    // Set random seed for reproducability
    unsafe { srand(21211411) }

    // Positions of CNOTs in each circuit
    let qft3 = vec![1, 0, 1, 0, 1, 0, 1];
    let qft5 = vec![
        0, 3, 0, 2, 2, 1, 2, 0, 1, 2, 3, 0, 0, 1, 2, 3, 1, 2, 3, 0, 1, 1, 2, 1, 2, 0, 2, 3, 0, 0, 1,
    ];
    let mul = vec![2, 1, 0, 1, 2, 3, 1, 2, 1, 2, 0, 0, 1, 1];
    let fredkin = vec![0, 1, 1, 0, 1, 1, 0, 1];
    let qaoa = vec![
        0, 2, 3, 1, 3, 0, 3, 2, 2, 2, 3, 2, 1, 0, 1, 1, 2, 1, 2, 3, 0, 3, 0, 1, 2, 1, 0,
    ];
    let hhl = vec![1, 0, 1, 0];
    let peres = vec![1, 0, 1, 0, 0, 1, 0];
    let grover3 = vec![1, 0, 1, 0, 0, 1, 0];
    let tfim_5_100 = vec![1, 3, 0, 2, 3, 1, 2, 2, 3, 1, 3, 0, 2, 3, 1, 2, 3, 0, 0, 1];
    let toffoli = vec![0, 1, 0, 1, 0, 0, 1, 0];
    let tfim_6_1 = vec![0, 2, 3, 4, 0, 2, 4, 1, 3, 1];
    let adder = vec![1, 2, 1, 0, 1, 0, 1, 0, 1, 2, 1, 2, 0, 1];
    let qft4 = vec![0, 2, 1, 2, 0, 2, 1, 2, 0, 1, 0, 1, 2, 1, 2, 0];
    let or = vec![1, 0, 1, 0, 0, 1, 0, 0];
    let tfim_4_95 = vec![1, 2, 0, 2, 1, 1, 2, 0, 0, 1, 2, 2];
    let vqe = vec![
        0, 2, 0, 0, 1, 0, 0, 1, 1, 1, 2, 0, 0, 1, 2, 0, 2, 1, 2, 0, 2, 1, 1, 0,
    ];
    let hlf = vec![1, 3, 0, 3, 0, 1, 1, 2, 2, 3, 1, 2, 0, 2, 1];

    let qft3_bytes = include_bytes!("unitaries/qft3.npy");
    let qft5_bytes = include_bytes!("unitaries/qft5.npy");
    let mul_bytes = include_bytes!("unitaries/mul.npy");
    let fredkin_bytes = include_bytes!("unitaries/fredkin.npy");
    let qaoa_bytes = include_bytes!("unitaries/qaoa.npy");
    let hhl_bytes = include_bytes!("unitaries/hhl.npy");
    let peres_bytes = include_bytes!("unitaries/peres.npy");
    let grover3_bytes = include_bytes!("unitaries/grover3.npy");
    let tfim_5_100_bytes = include_bytes!("unitaries/tfim-5-100.npy");
    let toffoli_bytes = include_bytes!("unitaries/toffoli.npy");
    let tfim_6_1_bytes = include_bytes!("unitaries/tfim-6-1.npy");
    let adder_bytes = include_bytes!("unitaries/adder.npy");
    let qft4_bytes = include_bytes!("unitaries/qft4.npy");
    let or_bytes = include_bytes!("unitaries/or.npy");
    let tfim_4_95_bytes = include_bytes!("unitaries/tfim-4-95.npy");
    let vqe_bytes = include_bytes!("unitaries/vqe.npy");
    let hlf_bytes = include_bytes!("unitaries/hlf.npy");

    let benchs = vec![
        ("qft3", qft3, qft3_bytes.to_vec(), 3),
        ("fredkin", fredkin, fredkin_bytes.to_vec(), 3),
        ("hhl", hhl, hhl_bytes.to_vec(), 3),
        ("peres", peres, peres_bytes.to_vec(), 3),
        ("grover3", grover3, grover3_bytes.to_vec(), 3),
        ("toffoli", toffoli, toffoli_bytes.to_vec(), 3),
        ("or", or, or_bytes.to_vec(), 3),
        ("adder", adder, adder_bytes.to_vec(), 4),
        ("qft4", qft4, qft4_bytes.to_vec(), 4),
        ("tfim_4_95", tfim_4_95, tfim_4_95_bytes.to_vec(), 4),
        ("tfim_5_100", tfim_5_100, tfim_5_100_bytes.to_vec(), 5),
        ("qft5", qft5, qft5_bytes.to_vec(), 5),
        ("qaoa", qaoa, qaoa_bytes.to_vec(), 5),
        ("mul", mul, mul_bytes.to_vec(), 5),
        ("vqe", vqe, vqe_bytes.to_vec(), 5),
        ("hlf", hlf, hlf_bytes.to_vec(), 5),
        ("tfim_6_1", tfim_6_1, tfim_6_1_bytes.to_vec(), 6),
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
        group.sample_size(10).bench_function(name, |b| {
            b.iter(|| optimize_qfactor(&instantiator, &mut circ.clone(), target.clone(), &x0))
        });
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
        group.sample_size(10).bench_function(name.clone(), |b| {
            b.iter(|| {
                optimize_ceres(
                    &minimizer,
                    &ResidualFunction::HilbertSchmidt(Box::new(cost.clone())),
                    &x0,
                )
            })
        });
    }
}

criterion_group!(ceres, bench_ceres);
criterion_group!(qfactor, bench_qfactor);
criterion_main!(ceres, qfactor);
