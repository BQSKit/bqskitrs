use crate::Circuit;

#[cfg(feature = "bfgs")]
mod bfgs;
#[cfg(feature = "ceres")]
mod ceres;

#[cfg(feature = "bfgs")]
pub use self::bfgs::BfgsJacSolver;
#[cfg(feature = "ceres")]
pub use self::ceres::CeresJacSolver;

#[cfg(feature = "bfgs")]
mod cost_fn;
#[cfg(feature = "ceres")]
mod residual_fn;

#[cfg(feature = "bfgs")]
pub use cost_fn::*;
#[cfg(feature = "ceres")]
pub use residual_fn::*;

use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait Minimizer {
    type CostFunctionTy: CostFn;
    fn minimize(&self, cost_fn: Self::CostFunctionTy, x0: Vec<f64>) -> Vec<f64>;
}
