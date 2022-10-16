use ndarray::{s, Array2, Zip};
use ndarray_linalg::c64;
pub trait SwapRows {
    fn swap_rows(&mut self, idx_a: usize, idx_b: usize);
}

impl SwapRows for Array2<c64> {
    fn swap_rows(&mut self, idx_a: usize, idx_b: usize) {
        let (row_a, row_b) = self.multi_slice_mut((s![idx_a, ..], s![idx_b, ..]));
        Zip::from(row_a).and(row_b).for_each(std::mem::swap);
    }
}
