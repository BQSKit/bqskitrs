use ndarray::{Array2, ArrayD, ArrayView2, Ix2};
use ndarray_linalg::c64;

use crate::utils::{argsort, trace};
use crate::squaremat::*;
use itertools::Itertools;

/// A type to build unitaries using tensor networks
pub struct UnitaryBuilder {
    pub num_qudits: usize,
    pub num_idxs: usize,
    pub dim: usize,
    pub pi: Vec<usize>,
    pub radixes: Vec<usize>,
    pub tensor: Option<ArrayD<c64>>,
}

impl UnitaryBuilder {
    pub fn new(num_qudits: usize, radixes: Vec<usize>) -> Self {
        let dim = radixes.iter().product();
        let num_idxs = num_qudits * 2;
        let pi = Vec::from_iter(0..num_idxs);
        let mut tensor = Array2::<c64>::eye(dim).into_dyn();
        tensor = tensor
            .into_shape([&radixes[..], &radixes[..]].concat())
            .unwrap();
        UnitaryBuilder {
            num_qudits,
            num_idxs,
            dim,
            pi,
            radixes,
            tensor: Some(tensor),
        }
    }

    pub fn get_utry(&mut self) -> Array2<c64> {
        self.reset_idxs();
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

    pub fn permute_idxs(&mut self, pi: Vec<usize>) {
        let revert: Vec<usize> = self.pi.iter()
            .enumerate()
            .sorted_by(|(_idx_a, a), (_idx_b, b)| a.cmp(b))
            .map(|(idx, _a)| idx)
            .collect();
        let composed: Vec<usize> = pi.iter().map(|&x| revert[x]).collect();
        self.tensor = Some(self.tensor.take().unwrap().permuted_axes(composed));
        self.pi = pi;
    }

    pub fn reset_idxs(&mut self) {
        self.permute_idxs(Vec::from_iter(0..self.num_idxs));
    }

    pub fn get_current_shape(&self) -> Vec<usize> {
        self.pi.iter().map(
            |&x|
            match x {
                r if r < self.num_qudits => self.radixes[x],
                r if r >= self.num_qudits => self.radixes[x - self.num_qudits],
                _ => panic!("Tensor index is invalid."),
            }
        ).collect()
    }

    pub fn apply_right(&mut self, utry: ArrayView2<c64>, location: &[usize], inverse: bool) {
        // Permute Tensor Indicies
        let left_perm = location.iter();
        let right_perm = (0..self.num_idxs).filter(|x| !location.contains(&x));
        let mut perm = vec![];
        perm.extend(left_perm);
        perm.extend(right_perm);
        self.permute_idxs(perm);

        // Reshape
        let owned_tensor = self.tensor.take().unwrap();
        let shape = self.get_current_shape();
        let left_dim: usize = shape[..location.len()].iter().product();
        let reshaped = owned_tensor
            .to_shape((left_dim, self.dim * self.dim / left_dim))
            .expect("Cannot reshape tensor to matrix");
        
        // Apply Unitary
        let prod = utry.dot(&reshaped.view());
        let reshape_back = prod
            .into_shape(shape)
            .expect("Failed to reshape matrix product back");
        self.tensor = Some(reshape_back.to_owned());
    }

    pub fn apply_left(&mut self, utry: ArrayView2<c64>, location: &[usize], inverse: bool) {
        // Permute Tensor Indicies
        let right_perm: Vec<usize> = location.iter().map(|x| x + self.num_qudits).collect();
        let left_perm = (0..self.num_idxs).filter(|x| !right_perm.contains(&x));
        let mut perm = vec![];
        perm.extend(left_perm);
        perm.extend(right_perm);
        self.permute_idxs(perm);

        // Reshape
        let owned_tensor = self.tensor.take().unwrap();
        // let shape = owned_tensor.shape().clone();
        let shape = self.get_current_shape();
        let right_dim: usize = shape[shape.len()-location.len()..].iter().product();
        let reshaped = owned_tensor
            .to_shape((self.dim * self.dim / right_dim, right_dim))
            .expect("Cannot reshape tensor to matrix");
        
        // Apply Unitary
        let prod = reshaped.dot(&utry);
        let reshape_back = prod
            .into_shape(shape)
            .expect("Failed to reshape matrix product back");
        self.tensor = Some(reshape_back.to_owned());
    }

    pub fn calc_env_matrix(&mut self, location: &[usize]) -> Array2<c64> {
        self.reset_idxs();
        let mut left_perm: Vec<usize> = (0..self.num_qudits).filter(|x| !location.contains(x)).collect();
        let left_perm_copy = left_perm.clone();
        let left_extension = left_perm_copy.iter().map(|x| x + self.num_qudits);
        left_perm.extend(left_extension);
        let mut right_perm = location.to_owned();
        right_perm.extend(location.iter().map(|x| x + self.num_qudits));

        let mut perm = vec![];
        perm.append(&mut left_perm);
        perm.append(&mut right_perm);
        let a = match &self.tensor {
            Some(t) => t.clone().permuted_axes(perm),
            None => panic!("Tensor was unexpectedly None."),
        };
        let reshaped = a
            .to_shape([
                2usize.pow(self.num_qudits as u32 - location.len() as u32),
                2usize.pow(self.num_qudits as u32 - location.len() as u32),
                2usize.pow(location.len() as u32),
                2usize.pow(location.len() as u32),
            ])
            .expect("Failed to reshape in calc_env_matrix.");
        trace(reshaped.view())
    }
}
