use ceres::CeresSolver;
use ndarray::Array2;
use rand::{thread_rng, Rng};
use squaremat::SquareMatrix;

use crate::utils::{matrix_residuals, matrix_residuals_jac};

use std::f64::consts::PI;

use super::Solver;

use crate::gates::{Gradient, Unitary};
use crate::Circuit;

pub struct CeresJacSolver {
    solver: CeresSolver,
}

impl CeresJacSolver {
    pub fn new(num_threads: usize, ftol: f64, gtol: f64) -> Self {
        CeresJacSolver {
            solver: CeresSolver::new(num_threads, ftol, gtol),
        }
    }
}

impl Solver for CeresJacSolver {
    fn solve_for_unitary(
        &self,
        circ: &Circuit,
        unitary: &SquareMatrix,
        x0: Option<Vec<f64>>,
    ) -> (SquareMatrix, Vec<f64>) {
        let i = circ.num_params();
        let mut rng = thread_rng();
        let mut x0: Vec<f64> = if let Some(x) = x0 {
            x
        } else {
            (0..i).map(|_| rng.gen_range(0.0..2.0 * PI)).collect()
        };
        let eye = Array2::eye(unitary.size);
        let mut cost_fn = |params: &[f64], resids: &mut [f64], jac: Option<&mut [f64]>| {
            let (m, jacs) = circ.get_utry_and_grad(&params, &circ.constant_gates);
            let res = matrix_residuals(&unitary, &m, &eye);
            resids.copy_from_slice(&res);
            if let Some(jacobian) = jac {
                let jac_mat = matrix_residuals_jac(&unitary, &m, &jacs);
                let v: Vec<f64> = jac_mat.iter().copied().collect();
                jacobian.copy_from_slice(&v)
            }
        };
        self.solver.solve(
            &mut cost_fn,
            &mut x0,
            unitary.size * unitary.size * 2,
            100 * i,
        );
        (circ.get_utry(&x0, &circ.constant_gates), x0)
    }
}
