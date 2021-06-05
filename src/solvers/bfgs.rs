use super::Solver;
use crate::gates::{Gradient, Unitary};
use crate::Circuit;

use crate::utils::{matrix_distance_squared, matrix_distance_squared_jac};
use nlopt::*;
use rand::{thread_rng, Rng};
use squaremat::SquareMatrix;
use std::f64::consts::PI;

pub struct BfgsJacSolver {
    size: usize,
}

impl BfgsJacSolver {
    pub fn new(size: usize) -> Self {
        BfgsJacSolver { size }
    }
}

impl Solver for BfgsJacSolver {
    fn solve_for_unitary(
        &self,
        circ: &Circuit,
        unitary: &SquareMatrix,
        x0: Option<Vec<f64>>,
    ) -> (SquareMatrix, Vec<f64>) {
        let i = circ.num_params();
        let f = |x: &[f64], gradient: Option<&mut [f64]>, _user_data: &mut ()| -> f64 {
            let dsq;
            if let Some(grad) = gradient {
                let (m, jac) = circ.get_utry_and_grad(&x, &circ.constant_gates);
                let (d, j) = matrix_distance_squared_jac(&unitary, &m, jac);
                dsq = d;
                grad.copy_from_slice(&j);
            } else {
                let m = circ.get_utry(&x, &circ.constant_gates);
                dsq = matrix_distance_squared(&unitary, &m);
            }
            dsq
        };
        let mut rng = thread_rng();
        let mut x0: Vec<f64> = if let Some(x) = x0 {
            x
        } else {
            (0..i).map(|_| rng.gen_range(0.0..2.0 * PI)).collect()
        };
        let mut fmin = Nlopt::new(Algorithm::Lbfgs, i, &f, Target::Minimize, ());
        fmin.set_upper_bound(2.0 * PI).unwrap();
        fmin.set_lower_bound(0.0).unwrap();
        fmin.set_stopval(1e-16).unwrap();
        fmin.set_maxeval(15000).unwrap();
        fmin.set_vector_storage(Some(self.size)).unwrap();
        match fmin.optimize(&mut x0) {
            Err((nlopt::FailState::Failure, _)) | Err((nlopt::FailState::RoundoffLimited, _)) => (),
            Ok(_) => (),
            Err(e) => panic!("Failed optimization! ({:?}, {})", e.0, e.1),
        }
        (circ.get_utry(&x0, &circ.constant_gates), x0)
    }
}
