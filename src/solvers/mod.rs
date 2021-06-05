use crate::Circuit;

#[cfg(feature = "bfgs")]
mod bfgs;
#[cfg(feature = "ceres")]
mod ceres;

#[cfg(feature = "bfgs")]
pub use self::bfgs::BfgsJacSolver;
#[cfg(feature = "ceres")]
pub use self::ceres::CeresJacSolver;

use enum_dispatch::enum_dispatch;
use squaremat::SquareMatrix;

#[enum_dispatch]
pub trait Solver {
    fn solve_for_unitary(
        &self,
        circ: &Circuit,
        unitary: &SquareMatrix,
        x0: Option<Vec<f64>>,
    ) -> (SquareMatrix, Vec<f64>);
}

#[enum_dispatch(Solver)]
pub enum Minimizer {
    #[cfg(feature = "bfgs")]
    Bfgs(BfgsJacSolver),
    #[cfg(feature = "ceres")]
    Ceres(CeresJacSolver),
}
