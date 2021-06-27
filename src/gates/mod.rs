mod constant;
mod dynamic;
mod gradient;
mod optimize;
mod parameterized;
mod size;
mod unitary;
mod utils;

use std::rc::Rc;

pub use self::constant::ConstantGate;
pub use self::dynamic::DynGate;
pub use self::gradient::Gradient;
pub use self::optimize::Optimize;
pub use self::parameterized::*;
pub use self::size::Size;
pub use self::unitary::Unitary;

use ndarray::ArrayViewMut2;
use num_complex::Complex64;
use squaremat::SquareMatrix;

use derive_more::From;

#[derive(Clone, Debug, From)]
pub enum Gate {
    Constant(ConstantGate),
    U1(U1Gate),
    U2(U2Gate),
    U3(U3Gate),
    RX(RXGate),
    RY(RYGate),
    RZ(RZGate),
    VariableUnitary(VariableUnitaryGate),
    Dynamic(Rc<dyn DynGate>),
}

impl Unitary for Gate {
    fn num_params(&self) -> usize {
        match self {
            Gate::Constant(_) => 0,
            Gate::U1(_) => 1,
            Gate::U2(_) => 2,
            Gate::U3(_) => 3,
            Gate::RX(_) => 1,
            Gate::RY(_) => 1,
            Gate::RZ(_) => 1,
            Gate::VariableUnitary(v) => v.num_params(),
            Gate::Dynamic(d) => d.num_params(),
        }
    }

    fn get_utry(&self, params: &[f64], const_gates: &[SquareMatrix]) -> SquareMatrix {
        match self {
            Gate::Constant(c) => c.get_utry(params, const_gates),
            Gate::U1(u) => u.get_utry(params, const_gates),
            Gate::U2(u) => u.get_utry(params, const_gates),
            Gate::U3(u) => u.get_utry(params, const_gates),
            Gate::RX(x) => x.get_utry(params, const_gates),
            Gate::RY(y) => y.get_utry(params, const_gates),
            Gate::RZ(z) => z.get_utry(params, const_gates),
            Gate::VariableUnitary(v) => v.get_utry(params, const_gates),
            Gate::Dynamic(d) => d.get_utry(params, const_gates),
        }
    }
}

impl Gradient for Gate {
    fn get_grad(&self, params: &[f64], const_gates: &[SquareMatrix]) -> Vec<SquareMatrix> {
        match self {
            Gate::Constant(c) => c.get_grad(params, const_gates),
            Gate::U1(u) => u.get_grad(params, const_gates),
            Gate::U2(u) => u.get_grad(params, const_gates),
            Gate::U3(u) => u.get_grad(params, const_gates),
            Gate::RX(x) => x.get_grad(params, const_gates),
            Gate::RY(y) => y.get_grad(params, const_gates),
            Gate::RZ(z) => z.get_grad(params, const_gates),
            Gate::VariableUnitary(v) => v.get_grad(params, const_gates),
            Gate::Dynamic(d) => d.get_grad(params, const_gates),
        }
    }
}

impl Size for Gate {
    fn get_size(&self) -> usize {
        match self {
            Gate::Constant(c) => c.get_size(),
            Gate::U1(_) => 1,
            Gate::U2(_) => 1,
            Gate::U3(_) => 1,
            Gate::RX(_) => 1,
            Gate::RY(_) => 1,
            Gate::RZ(_) => 1,
            Gate::VariableUnitary(v) => v.get_size(),
            Gate::Dynamic(d) => d.get_size(),
        }
    }
}

impl Optimize for Gate {
    fn optimize(&self, env_matrix: ArrayViewMut2<Complex64>) -> Vec<f64> {
        match self {
            Gate::Constant(_) => todo!(),
            Gate::U1(u) => u.optimize(env_matrix),
            Gate::U2(u) => u.optimize(env_matrix),
            Gate::U3(u) => u.optimize(env_matrix),
            Gate::RX(x) => x.optimize(env_matrix),
            Gate::RY(y) => y.optimize(env_matrix),
            Gate::RZ(z) => z.optimize(env_matrix),
            Gate::VariableUnitary(v) => v.optimize(env_matrix),
            Gate::Dynamic(d) => d.optimize(env_matrix),
        }
    }
}
