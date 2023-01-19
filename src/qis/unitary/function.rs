use enum_dispatch::enum_dispatch;
use ndarray::Array2;
use ndarray_linalg::c64;

#[enum_dispatch]
pub trait UnitaryFunction {
    // fn get_num_params(&self) -> usize;
    // fn get_num_qudits(&self) -> usize;
    // fn get_radixes(&self) -> Vec<usize>;
    // fn get_dim(&self) -> usize
    // {
    //     return self.get_radixes().iter().product()
    // }
    fn get_utry(&self, params: &[f64]) -> Array2<c64>;
}