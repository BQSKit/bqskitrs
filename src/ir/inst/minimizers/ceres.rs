use ceres::CeresSolver;

use super::{DifferentiableResidualFn, Minimizer, ResidualFn, ResidualFunction};

pub struct CeresJacSolver {
    solver: CeresSolver,
}

impl CeresJacSolver {
    pub fn new(num_threads: usize, ftol: f64, gtol: f64, report: bool) -> Self {
        CeresJacSolver {
            solver: CeresSolver::new(num_threads, ftol, gtol, report),
        }
    }
}

impl Minimizer for CeresJacSolver {
    type CostFunctionTy = ResidualFunction;
    fn minimize(&self, cost_fn: &Self::CostFunctionTy, x0: &[f64]) -> Vec<f64> {
        if x0.is_empty() {
            return x0.to_vec();
        }
        let i = x0.len();
        let mut cost_fun = |params: &[f64], resids: &mut [f64], jac: Option<&mut [f64]>| {
            let (res, jacs) = cost_fn.get_residuals_and_grad(&params);
            resids.copy_from_slice(&res);
            if let Some(jacobian) = jac {
                let jac_mat = jacs;
                let v: Vec<f64> = jac_mat.iter().copied().collect();
                jacobian.copy_from_slice(&v)
            }
        };
        let mut x = x0.to_vec();
        self.solver
            .solve(&mut cost_fun, &mut x, cost_fn.num_residuals(), 100 * i);
        x
    }
}
