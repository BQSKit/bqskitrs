use squaremat::SquareMatrix;

use crate::solvers::{Minimizer, Solver};
use crate::Circuit;

/// Instantiate circuit to solve for target
pub struct Instantiator {
    minimizer: Minimizer,
}

impl Instantiator {
    fn instantiate(
        &self,
        circ: &Circuit,
        target: &SquareMatrix,
        x0: Option<Vec<f64>>,
    ) -> SquareMatrix {
        self.minimizer.solve_for_unitary(circ, target, x0).0
    }
}
