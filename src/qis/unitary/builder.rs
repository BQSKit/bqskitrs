use ndarray::{Array2, ArrayD, ArrayView2, Ix2};
use ndarray_linalg::c64;

use crate::utils::{argsort, trace};
use crate::squaremat::*;

/// A type to build unitaries using tensor networks
pub struct UnitaryBuilder {
    size: usize,
    radixes: Vec<usize>,
    tensor: Option<ArrayD<c64>>,
    dim: usize,
}

impl UnitaryBuilder {
    pub fn new(size: usize, radixes: Vec<usize>) -> Self {
        let dim = radixes.iter().product();
        let mut tensor = Array2::<c64>::eye(dim).into_dyn();
        tensor = tensor
            .into_shape([&radixes[..], &radixes[..]].concat())
            .unwrap();
        UnitaryBuilder {
            size,
            radixes,
            tensor: Some(tensor),
            dim,
        }
    }

    pub fn get_utry(&self) -> Array2<c64> {
        match &self.tensor {
            Some(t) => t
                .to_shape((self.dim, self.dim))
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap()
                .to_owned(),
            None => panic!("Tensor was unexpectedly None."),
        }
    }

    pub fn apply_right(&mut self, utry: ArrayView2<c64>, location: &[usize], inverse: bool) {
        let left_perm = location.iter();
        let mid_perm = (0..self.size).filter(|x| !location.contains(&x));
        let right_perm = (0..self.size).map(|x| x + self.size);

        let left_dim: usize = left_perm.clone().map(|i| self.radixes[*i]).product();
        let unitary = if inverse {
            let conj = utry.conj();
            conj.reversed_axes()
        } else {
            utry.to_owned()
        };
        let mut perm = vec![];
        perm.extend(left_perm);
        perm.extend(mid_perm);
        perm.extend(right_perm);

        let permuted = self.tensor.take().unwrap().permuted_axes(perm.clone());
        let dim: usize = permuted.shape().iter().product();
        let reshaped = permuted
            .to_shape((left_dim, dim / left_dim))
            .expect("Cannot reshape tensor to matrix");
        let prod = unitary.matmul(reshaped.view());

        let radixes = [&self.radixes[..], &self.radixes[..]].concat();
        let shape: Vec<usize> = perm.iter().map(|p| radixes[*p]).collect();
        let reshape_back = prod
            .into_shape(shape)
            .expect("Failed to reshape matrix product back");
        self.tensor = Some(reshape_back.permuted_axes(argsort(perm)).into_dyn());
    }

    pub fn apply_left(&mut self, utry: ArrayView2<c64>, location: &[usize], inverse: bool) {
        let left_perm = 0..self.size;
        let mid_perm = left_perm.clone().filter_map(|x| {
            if location.contains(&x) {
                None
            } else {
                Some(x + self.size)
            }
        });
        let right_perm = location.iter().map(|x| x + self.size);

        let right_dim: usize = right_perm
            .clone()
            .map(|i| self.radixes[i - self.size])
            .product();

        let unitary = if inverse {
            utry.conj().reversed_axes()
        } else {
            utry.to_owned()
        };
        let mut perm = vec![];
        perm.extend(left_perm);
        perm.extend(mid_perm);
        perm.extend(right_perm);

        let permuted = self.tensor.take().unwrap().permuted_axes(perm.clone());
        let dim: usize = permuted.shape().iter().product();
        let reshaped = permuted
            .to_shape((dim / right_dim, right_dim))
            .expect("Cannot reshape tensor to matrix");
        let prod = reshaped.view().matmul(unitary.view());
        let radixes = [&self.radixes[..], &self.radixes[..]].concat();
        let shape: Vec<usize> = perm.iter().map(|p| radixes[*p]).collect();
        let reshape_back = prod
            .into_shape(shape)
            .expect("Failed to reshape matrix product back");
        self.tensor = Some(reshape_back.permuted_axes(argsort(perm)).into_dyn());
    }

    pub fn eval_apply_right(&self, m: ArrayView2<c64>, location: &[usize]) -> Array2<c64> {
        let left_perm = location.iter();
        let mid_perm = (0..self.size).filter(|x| !location.contains(&x));
        let right_perm = (0..self.size).map(|x| x + self.size);

        let left_dim: usize = left_perm.clone().map(|i| self.radixes[*i]).product();

        let matrix = m.to_owned();

        let mut perm = vec![];
        perm.extend(left_perm);
        perm.extend(mid_perm);
        perm.extend(right_perm);

        let mut tensor_copy = self.tensor.clone();
        let permuted = tensor_copy.take().unwrap().permuted_axes(perm.clone());
        let dim: usize = permuted.shape().iter().product();
        let reshaped = permuted
            .to_shape((left_dim, dim / left_dim))
            .expect("Cannot reshape tensor to matrix");
        let prod = matrix.matmul(reshaped.view());

        let radixes = [&self.radixes[..], &self.radixes[..]].concat();
        let shape: Vec<usize> = perm.iter().map(|p| radixes[*p]).collect();
        let reshape_back = prod
            .into_shape(shape)
            .expect("Failed to reshape matrix product back");
        let eval_tensor = reshape_back.permuted_axes(argsort(perm)).into_dyn();
        let matrix_dim = self.dim.clone();
        let eval_matrix = eval_tensor
            .to_shape((matrix_dim, matrix_dim))
            .expect("Cannot reshape tensor to matrix");
        eval_matrix.to_owned()
    }

    pub fn eval_apply_left(&self, m: ArrayView2<c64>, location: &[usize]) -> Array2<c64> {
        let left_perm = 0..self.size;
        let mid_perm = left_perm.clone().filter_map(|x| {
            if location.contains(&x) {
                None
            } else {
                Some(x + self.size)
            }
        });
        let right_perm = location.iter().map(|x| x + self.size);

        let right_dim: usize = right_perm
            .clone()
            .map(|i| self.radixes[i - self.size])
            .product();
        
        let mut perm = vec![];
        perm.extend(left_perm);
        perm.extend(mid_perm);
        perm.extend(right_perm);

        let mut tensor_copy = self.tensor.clone();
        let permuted = tensor_copy.take().unwrap().permuted_axes(perm.clone());
        let dim: usize = permuted.shape().iter().product();
        let reshaped = permuted
            .to_shape((dim / right_dim, right_dim))
            .expect("Cannot reshape tensor to matrix");
        let prod = reshaped.view().matmul(m.view());

        let radixes = [&self.radixes[..], &self.radixes[..]].concat();
        let shape: Vec<usize> = perm.iter().map(|p| radixes[*p]).collect();
        let reshape_back = prod
            .into_shape(shape)
            .expect("Failed to reshape matrix product back");
        let eval_tensor = reshape_back.permuted_axes(argsort(perm)).into_dyn();
        let matrix_dim = self.dim.clone();
        let eval_matrix = eval_tensor
            .to_shape((matrix_dim, matrix_dim))
            .expect("Cannot reshape tensor to matrix");
        eval_matrix.to_owned()
    }

    pub fn calc_env_matrix(&self, location: &[usize]) -> Array2<c64> {
        let mut left_perm: Vec<usize> = (0..self.size).filter(|x| !location.contains(x)).collect();
        let left_perm_copy = left_perm.clone();
        let left_extension = left_perm_copy.iter().map(|x| x + self.size);
        left_perm.extend(left_extension);
        let mut right_perm = location.to_owned();
        right_perm.extend(location.iter().map(|x| x + self.size));

        let mut perm = vec![];
        perm.append(&mut left_perm);
        perm.append(&mut right_perm);
        let a = match &self.tensor {
            Some(t) => t.clone().permuted_axes(perm),
            None => panic!("Tensor was unexpectedly None."),
        };
        let reshaped = a
            .to_shape([
                2usize.pow(self.size as u32 - location.len() as u32),
                2usize.pow(self.size as u32 - location.len() as u32),
                2usize.pow(location.len() as u32),
                2usize.pow(location.len() as u32),
            ])
            .expect("Failed to reshape in calc_env_matrix.");
        trace(reshaped.view())
    }
}
