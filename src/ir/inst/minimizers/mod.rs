mod bfgs;
mod ceres;

pub use self::bfgs::BfgsJacSolver;
pub use self::ceres::CeresJacSolver;

mod cost_fn;
mod residual_fn;

pub use cost_fn::*;
pub use residual_fn::*;

use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait Minimizer {
    type CostFunctionTy: CostFn;
    fn minimize(&self, cost_fn: &Self::CostFunctionTy, x0: &[f64]) -> Vec<f64>;
}
