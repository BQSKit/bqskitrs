use ndarray::Array2;
use num_complex::Complex64;

use ndarray_linalg::trace::Trace;

use super::Instantiate;
use crate::gates::Optimize;
use crate::{circuit::Circuit, gates::Unitary, unitary_builder::UnitaryBuilder};

#[derive(Clone, Copy)]
pub struct QFactorInstantiator {
    diff_tol_a: f64,
    diff_tol_r: f64,
    dist_tol: f64,
    max_iters: usize,
    min_iters: usize,
    // slowdown_factor: f64, // TODO
    reinit_delay: usize,
}

impl Default for QFactorInstantiator {
    fn default() -> Self {
        QFactorInstantiator {
            diff_tol_a: 1e-12,
            diff_tol_r: 1e-6,
            dist_tol: 1e-16,
            max_iters: 100000,
            min_iters: 1000,
            //slowdown_factor: 0.0,
            reinit_delay: 40,
        }
    }
}

impl QFactorInstantiator {
    pub fn new(
        diff_tol_a: Option<f64>,
        diff_tol_r: Option<f64>,
        dist_tol: Option<f64>,
        max_iters: Option<usize>,
        min_iters: Option<usize>,
        _slowdown_factor: Option<f64>,
        reinit_delay: Option<usize>,
    ) -> Self {
        QFactorInstantiator {
            diff_tol_a: diff_tol_a.unwrap_or(1e-12),
            diff_tol_r: diff_tol_r.unwrap_or(1e-6),
            dist_tol: dist_tol.unwrap_or(1e-16),
            max_iters: max_iters.unwrap_or(100000),
            min_iters: min_iters.unwrap_or(1000),
            //slowdown_factor: slowdown_factor.unwrap_or(0.0),
            reinit_delay: reinit_delay.unwrap_or(40),
        }
    }

    pub fn initialize_circuit_tensor(
        &self,
        circuit: &Circuit,
        target: &Array2<Complex64>,
    ) -> UnitaryBuilder {
        let mut unitary_builder = UnitaryBuilder::new(circuit.size, circuit.radixes.clone());
        let entire_circ_location: Vec<usize> = (0..circuit.size).collect();
        unitary_builder.apply_right(target.view(), &entire_circ_location, true);
        unitary_builder.apply_right(
            circuit.get_utry(&[], &circuit.constant_gates).view(),
            &entire_circ_location,
            false,
        );
        unitary_builder
    }

    pub fn sweep_circuit(&self, unitary_builder: &mut UnitaryBuilder, circuit: &mut Circuit) {
        // Start by looping backwards
        for op in circuit.ops.iter_mut().rev() {
            let gate = op.get_utry(&[], &circuit.constant_gates);
            unitary_builder.apply_right(gate.view(), &op.location, true);
            if op.num_params() != 0 {
                let mut env = unitary_builder.calc_env_matrix(&op.location);
                let params = op.optimize(env.view_mut());
                op.params = params;
            }
            let gate = op.get_utry(&[], &circuit.constant_gates);
            unitary_builder.apply_left(gate.view(), &op.location, false);
        }

        // reset for new loop through all the gates the opposite order
        for op in circuit.ops.iter_mut() {
            let gate = op.get_utry(&[], &circuit.constant_gates);
            unitary_builder.apply_left(gate.view(), &op.location, true);

            if op.num_params() != 0 {
                let mut env = unitary_builder.calc_env_matrix(&op.location);
                let params = op.optimize(env.view_mut());
                op.params = params;
            }
            let gate = op.get_utry(&[], &circuit.constant_gates);
            unitary_builder.apply_right(gate.view(), &op.location, false);
        }
    }
}

impl Instantiate for QFactorInstantiator {
    fn instantiate(
        &self,
        circuit: &mut Circuit,
        target: Array2<Complex64>,
        x0: &[f64],
    ) -> Vec<f64> {
        if x0.len() != circuit.num_params() {
            panic!(
                "Too few parameters in x0 for the QFactor instantiator, expected {}, got {}",
                circuit.num_params(),
                x0.len()
            );
        }
        circuit.set_params(x0);

        let mut unitary_builder = self.initialize_circuit_tensor(&circuit, &target);
        let mut dist1 = 0.0f64;
        let mut dist2 = 0.0f64;

        let mut it = 0usize;

        loop {
            if it > self.min_iters {
                let diff_tol = self.diff_tol_a + self.diff_tol_r * dist1.abs();
                if (dist1 - dist2).abs() <= diff_tol {
                    break;
                }

                if it > self.max_iters {
                    break;
                }
            }

            it += 1;

            self.sweep_circuit(&mut unitary_builder, circuit);

            dist2 = dist1;
            dist1 = unitary_builder.get_utry().trace().unwrap().norm();
            dist1 = 1. - (dist1 / 2f64.powi(circuit.size as i32));

            if dist1 < self.dist_tol {
                return circuit.get_params();
            }

            if it % self.reinit_delay == 0 {
                unitary_builder = self.initialize_circuit_tensor(&circuit, &target)
            }
        }
        circuit.get_params()
    }
}
