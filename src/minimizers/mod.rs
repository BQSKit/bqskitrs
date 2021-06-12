#[cfg(feature = "bfgs")]
mod bfgs;
mod ceres;

#[cfg(feature = "bfgs")]
pub use self::bfgs::BfgsJacSolver;
pub use self::ceres::CeresJacSolver;

#[cfg(feature = "bfgs")]
mod cost_fn;
mod residual_fn;

#[cfg(feature = "bfgs")]
pub use cost_fn::*;
pub use residual_fn::*;

use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait Minimizer {
    type CostFunctionTy: CostFn;
    fn minimize(&self, cost_fn: Self::CostFunctionTy, x0: Vec<f64>) -> Vec<f64>;
}
