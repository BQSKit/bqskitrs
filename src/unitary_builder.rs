use ndarray::{Array2, ArrayD, Ix2};
use num_complex::Complex64;
use squaremat::SquareMatrix;

use crate::utils::{argsort, trace};

/// A type to build unitaries using tensor networks
pub struct UnitaryBuilder {
    size: usize,
    radixes: Vec<usize>,
    tensor: ArrayD<Complex64>,
    num_params: usize,
    dim: usize,
}

impl UnitaryBuilder {
    pub fn new(size: usize, radixes: Vec<usize>) -> Self {
        let dim = radixes.iter().product();
        UnitaryBuilder {
            size,
            radixes,
            tensor: Array2::<Complex64>::eye(2).into_dyn(),
            num_params: 0,
            dim,
        }
    }

    pub fn get_utry(&self) -> SquareMatrix {
        SquareMatrix::from_ndarray(
            self.tensor
                .to_shape((self.dim, self.dim))
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap()
                .to_owned(),
        )
    }

    pub fn apply_right(&mut self, utry: SquareMatrix, location: Vec<usize>, inverse: bool) {
        let mut left_perm = location.iter();
        let mut mid_perm = (0..self.size).filter(|x| !location.contains(&x));
        let mut right_perm = (0..self.size).map(|x| x + self.size);

        let left_dim: usize = left_perm.clone().map(|i| self.radixes[*i]).product();
        let unitary = if inverse {
            utry.H().into_ndarray()
        } else {
            utry.into_ndarray()
        };
        let mut perm = vec![];
        perm.extend(left_perm);
        perm.extend(mid_perm);
        perm.extend(right_perm);

        let permuted = self.tensor.clone().permuted_axes(perm.clone());
        let dim: usize = permuted.shape().iter().product();
        let reshaped = permuted
            .to_shape((left_dim, dim / left_dim))
            .expect("Cannot reshape tensor to matrix");
        let prod = unitary.dot(&reshaped);

        let radixes = [&self.radixes[..], &self.radixes[..]].concat();
        let shape: Vec<usize> = perm.iter().map(|p| radixes[*p]).collect();
        let reshape_back = prod
            .to_shape(shape)
            .expect("Failed to reshape matrix product back");
        self.tensor = reshape_back
            .permuted_axes(argsort(perm))
            .to_owned()
            .into_dyn();
    }

    pub fn apply_left(&mut self, utry: SquareMatrix, location: Vec<usize>, inverse: bool) {
        let left_perm = 0..self.size;
        let mid_perm = left_perm.clone().filter_map(|x| {
            if location.contains(&x) {
                None
            } else {
                Some(x + self.size)
            }
        });
        let right_perm = location.iter().map(|x| x + self.size);

        let right_dim: usize = location.iter().product();

        let unitary = if inverse {
            utry.H().into_ndarray()
        } else {
            utry.into_ndarray()
        };
        let mut perm = vec![];
        perm.extend(left_perm);
        perm.extend(mid_perm);
        perm.extend(right_perm);

        let permuted = self.tensor.clone().permuted_axes(perm.clone());
        let dim: usize = permuted.shape().iter().product();
        let reshaped = permuted
            .to_shape((dim / right_dim, right_dim))
            .expect("Cannot reshape tensor to matrix");
        let prod = unitary.dot(&reshaped);
        let radixes = [&self.radixes[..], &self.radixes[..]].concat();
        let shape: Vec<usize> = perm.iter().map(|p| radixes[*p]).collect();
        let reshape_back = prod
            .to_shape(shape)
            .expect("Failed to reshape matrix product back");
        self.tensor = reshape_back
            .permuted_axes(argsort(perm))
            .to_owned()
            .into_dyn();
    }

    pub fn calc_env_matrix(&self, location: Vec<usize>) -> Array2<Complex64> {
        let mut left_perm: Vec<usize> = (0..self.size).filter(|x| !location.contains(x)).collect();
        let left_perm_copy = left_perm.clone();
        let left_extension = left_perm_copy.iter().map(|x| x + self.size);
        left_perm.extend(left_extension);
        let mut right_perm = location.clone();
        right_perm.extend(location.iter().map(|x| x + self.size));

        let mut perm = vec![];
        perm.append(&mut left_perm);
        perm.append(&mut right_perm);
        let a = self.tensor.clone().permuted_axes(perm);
        let reshaped = a
            .to_shape([
                2usize.pow(self.size as u32 - location.len() as u32),
                2usize.pow(self.size as u32 - location.len() as u32),
                2usize.pow(location.len() as u32),
                2usize.pow(location.len() as u32),
            ])
            .expect("Failed to reshape in calc_env_matrix.")
            .to_owned();
        trace(reshaped)
    }
}
