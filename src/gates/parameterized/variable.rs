use crate::gates::{Gradient, Size};
use crate::gates::{Optimize, Unitary};

use ndarray::{Array2, ArrayViewMut2};
use num_complex::Complex64;
use squaremat::SquareMatrix;

use lapacke::{zgesdd, Layout};

fn svd(mut matrix: ArrayViewMut2<Complex64>) -> (Array2<Complex64>, Array2<Complex64>) {
    let size = matrix.shape()[0];
    let mut s = vec![0.0; size];
    let mut u = vec![Complex64::default(); size * size];
    let mut vt = vec![Complex64::default(); size * size];
    // safety: All matrices default to row major and have linear memory
    // s, u, vt all initialized above
    unsafe {
        zgesdd(
            Layout::RowMajor,
            b'a',
            size as i32,
            size as i32,
            matrix.as_slice_mut().unwrap(),
            size as i32,
            &mut s,
            &mut u,
            size as i32,
            &mut vt,
            size as i32,
        );
    }
    (
        Array2::from_shape_vec((size, size), u).unwrap(),
        Array2::from_shape_vec((size, size), vt).unwrap(),
    )
}

/// A variable n-qudit unitary gate
#[derive(Clone, Debug, PartialEq, Default)]
pub struct VariableUnitaryGate {
    size: usize,
    radixes: Vec<usize>,
    dim: usize,
    shape: (usize, usize),
    num_parameters: usize,
}

impl VariableUnitaryGate {
    pub fn new(size: usize, radixes: Vec<usize>) -> Self {
        let dim = radixes.iter().product();
        VariableUnitaryGate {
            size,
            radixes,
            dim,
            shape: (dim, dim),
            num_parameters: 2 * dim.pow(2u32),
        }
    }
}

impl Unitary for VariableUnitaryGate {
    fn num_params(&self) -> usize {
        self.num_parameters
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[SquareMatrix]) -> SquareMatrix {
        assert_eq!(self.num_params(), params.len());
        let mid = params.len() / 2;
        let (re, im) = params.split_at(mid);
        let vec: Vec<Complex64> = re
            .iter()
            .zip(im)
            .map(|(re, im)| Complex64::new(*re, *im))
            .collect();
        let len = vec.len();
        let mut matrix = Array2::from_shape_vec((self.dim, self.dim), vec)
            .unwrap_or_else(|_| panic!("Got vec of length {}, self.dim is {}", len, self.dim));
        let (u, vt) = svd(matrix.view_mut());
        SquareMatrix::from_ndarray(u.dot(&vt))
    }
}

impl Gradient for VariableUnitaryGate {
    fn get_grad(&self, _params: &[f64], _const_gates: &[SquareMatrix]) -> Vec<SquareMatrix> {
        unimplemented!()
    }
}

impl Size for VariableUnitaryGate {
    fn get_size(&self) -> usize {
        self.size
    }
}

impl Optimize for VariableUnitaryGate {
    fn optimize(&self, mut env_matrix: ArrayViewMut2<Complex64>) -> Vec<f64> {
        let (mut u, mut vt) = svd(env_matrix.view_mut());
        u.map_inplace(|i| *i = i.conj());
        vt.map_inplace(|i| *i = i.conj());
        let mat = vt.t().dot(&u.t());
        let mut ret = vec![0.0; self.num_parameters];
        for (i, cmplx) in mat.iter().enumerate() {
            ret[i % (self.num_parameters / 2)] = cmplx.re;
            ret[i % (self.num_parameters / 2) + (self.num_parameters / 2)] = cmplx.im;
        }
        ret
    }
}
