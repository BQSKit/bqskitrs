use enum_dispatch::enum_dispatch;
use ndarray::ArrayViewMut2;
use ndarray_linalg::c64;

#[enum_dispatch]
pub trait Optimize {
    fn optimize(&self, _env_matrix: ArrayViewMut2<c64>) -> Vec<f64> {
        unimplemented!()
    }
}
