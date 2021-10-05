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

use ndarray::{Array2, Array3, ArrayViewMut2};
use num_complex::Complex64;

use derive_more::From;

#[derive(Clone, Debug, From)]
pub enum Gate {
    Constant(ConstantGate),
    U1(U1Gate),
    U2(U2Gate),
    U3(U3Gate),
    U8(U8Gate),
    RX(RXGate),
    RY(RYGate),
    RZ(RZGate),
    RXX(RXXGate),
    RYY(RYYGate),
    RZZ(RZZGate),
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
            Gate::U8(_) => 8,
            Gate::RX(_) => 1,
            Gate::RY(_) => 1,
            Gate::RZ(_) => 1,
            Gate::RXX(_) => 1,
            Gate::RYY(_) => 1,
            Gate::RZZ(_) => 1,
            Gate::VariableUnitary(v) => v.num_params(),
            Gate::Dynamic(d) => d.num_params(),
        }
    }

    fn get_utry(&self, params: &[f64], const_gates: &[Array2<Complex64>]) -> Array2<Complex64> {
        match self {
            Gate::Constant(c) => c.get_utry(params, const_gates),
            Gate::U1(u) => u.get_utry(params, const_gates),
            Gate::U2(u) => u.get_utry(params, const_gates),
            Gate::U3(u) => u.get_utry(params, const_gates),
            Gate::U8(u) => u.get_utry(params, const_gates),
            Gate::RX(x) => x.get_utry(params, const_gates),
            Gate::RY(y) => y.get_utry(params, const_gates),
            Gate::RZ(z) => z.get_utry(params, const_gates),
            Gate::RXX(x) => x.get_utry(params, const_gates),
            Gate::RYY(y) => y.get_utry(params, const_gates),
            Gate::RZZ(z) => z.get_utry(params, const_gates),
            Gate::VariableUnitary(v) => v.get_utry(params, const_gates),
            Gate::Dynamic(d) => d.get_utry(params, const_gates),
        }
    }
}

impl Gradient for Gate {
    fn get_grad(&self, params: &[f64], const_gates: &[Array2<Complex64>]) -> Array3<Complex64> {
        match self {
            Gate::Constant(c) => c.get_grad(params, const_gates),
            Gate::U1(u) => u.get_grad(params, const_gates),
            Gate::U2(u) => u.get_grad(params, const_gates),
            Gate::U3(u) => u.get_grad(params, const_gates),
            Gate::U8(u) => u.get_grad(params, const_gates),
            Gate::RX(x) => x.get_grad(params, const_gates),
            Gate::RY(y) => y.get_grad(params, const_gates),
            Gate::RZ(z) => z.get_grad(params, const_gates),
            Gate::RXX(x) => x.get_grad(params, const_gates),
            Gate::RYY(y) => y.get_grad(params, const_gates),
            Gate::RZZ(z) => z.get_grad(params, const_gates),
            Gate::VariableUnitary(v) => v.get_grad(params, const_gates),
            Gate::Dynamic(d) => d.get_grad(params, const_gates),
        }
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        const_gates: &[Array2<Complex64>],
    ) -> (Array2<Complex64>, Array3<Complex64>) {
        match self {
            Gate::Constant(c) => c.get_utry_and_grad(params, const_gates),
            Gate::U1(u) => u.get_utry_and_grad(params, const_gates),
            Gate::U2(u) => u.get_utry_and_grad(params, const_gates),
            Gate::U3(u) => u.get_utry_and_grad(params, const_gates),
            Gate::U8(u) => u.get_utry_and_grad(params, const_gates),
            Gate::RX(x) => x.get_utry_and_grad(params, const_gates),
            Gate::RY(y) => y.get_utry_and_grad(params, const_gates),
            Gate::RZ(z) => z.get_utry_and_grad(params, const_gates),
            Gate::RXX(x) => x.get_utry_and_grad(params, const_gates),
            Gate::RYY(y) => y.get_utry_and_grad(params, const_gates),
            Gate::RZZ(z) => z.get_utry_and_grad(params, const_gates),
            Gate::VariableUnitary(v) => v.get_utry_and_grad(params, const_gates),
            Gate::Dynamic(d) => d.get_utry_and_grad(params, const_gates),
        }
    }
}

impl Size for Gate {
    fn num_qudits(&self) -> usize {
        match self {
            Gate::Constant(c) => c.num_qudits(),
            Gate::U1(_) => 1,
            Gate::U2(_) => 1,
            Gate::U3(_) => 1,
            Gate::U8(_) => 1,
            Gate::RX(_) => 1,
            Gate::RY(_) => 1,
            Gate::RZ(_) => 1,
            Gate::RXX(_) => 2,
            Gate::RYY(_) => 2,
            Gate::RZZ(_) => 2,
            Gate::VariableUnitary(v) => v.num_qudits(),
            Gate::Dynamic(d) => d.num_qudits(),
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
            Gate::U8(u) => u.optimize(env_matrix),
            Gate::RX(x) => x.optimize(env_matrix),
            Gate::RY(y) => y.optimize(env_matrix),
            Gate::RZ(z) => z.optimize(env_matrix),
            Gate::RXX(x) => x.optimize(env_matrix),
            Gate::RYY(y) => y.optimize(env_matrix),
            Gate::RZZ(z) => z.optimize(env_matrix),
            Gate::VariableUnitary(v) => v.optimize(env_matrix),
            Gate::Dynamic(d) => d.optimize(env_matrix),
        }
    }
}
