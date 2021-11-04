use super::{CostFunction, DifferentiableCostFn, Minimizer};

use crate::minimizers::CostFn;
use nlopt::*;

pub struct BfgsJacSolver {
    size: usize,
}

impl BfgsJacSolver {
    pub fn new(size: usize) -> Self {
        BfgsJacSolver { size }
    }
}

impl Minimizer for BfgsJacSolver {
    type CostFunctionTy = CostFunction;
    fn minimize(&self, cost_fn: &Self::CostFunctionTy, x0: &[f64]) -> Vec<f64> {
        if x0.is_empty() {
            return x0.to_vec();
        }
        let i = x0.len();
        let f = |x: &[f64], gradient: Option<&mut [f64]>, _user_data: &mut ()| -> f64 {
            let dsq;
            if let Some(grad) = gradient {
                let (d, j) = cost_fn.get_cost_and_grad(&x);
                dsq = d;
                grad.copy_from_slice(&j);
            } else {
                dsq = cost_fn.get_cost(&x);
            }
            dsq
        };
        let mut x = x0.to_vec();
        let mut fmin = Nlopt::new(Algorithm::Lbfgs, i, &f, Target::Minimize, ());
        fmin.set_stopval(1e-16).unwrap();
        fmin.set_maxeval(15000).unwrap();
        fmin.set_vector_storage(Some(self.size)).unwrap();
        match fmin.optimize(&mut x) {
            Err((nlopt::FailState::Failure, _)) | Err((nlopt::FailState::RoundoffLimited, _)) => (),
            Ok(_) => (),
            Err(e) => panic!("Failed optimization! ({:?}, {})", e.0, e.1),
        }
        x
    }
}
