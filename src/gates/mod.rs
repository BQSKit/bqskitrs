mod constant;
mod gradient;
mod parameterized;
mod size;
mod unitary;
mod utils;

pub use self::constant::ConstantGate;
pub use self::gradient::Gradient;
pub use self::parameterized::*;
pub use self::size::Size;
pub use self::unitary::Unitary;

use enum_dispatch::enum_dispatch;
use squaremat::SquareMatrix;

#[enum_dispatch(Unitary, Gradient, Size)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Gate {
    Constant(ConstantGate),
    U1(U1Gate),
    U2(U2Gate),
    U3(U3Gate),
    RX(RXGate),
    RY(RYGate),
    RZ(RZGate),
}