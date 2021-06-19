use enum_dispatch::enum_dispatch;
use ndarray::ArrayViewMut2;
use num_complex::Complex64;

#[enum_dispatch]
pub trait Optimize {
    fn optimize(&self, _env_matrix: ArrayViewMut2<Complex64>) -> Vec<f64> {
        unimplemented!()
    }
}
