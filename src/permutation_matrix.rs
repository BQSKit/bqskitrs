use std::{ops::MulAssign, vec};

use ndarray::{Array2, ArrayView2};
use ndarray_linalg::c64;
use crate::squaremat::*;

pub struct Permutation {
    perm: Vec<usize>,
}

impl Permutation {
    pub fn id(size: usize) -> Self {
        let perm = (0..size).collect();
        Permutation { perm }
    }

    pub fn new(perm: Vec<usize>) -> Self {
        Permutation { perm }
    }

    pub fn transpositions(&self) -> Vec<(usize, usize)> {
        let mut res = vec![];
        let a = self.cyclic_form();
        for mut x in a {
            let nx = x.len();
            match nx {
                2 => res.push((x[0], x[1])),
                d if d > 2 => {
                    let (start, end) = x.split_at_mut(1);
                    end.reverse();
                    let first = start[0];
                    for y in end {
                        res.push((first, *y));
                    }
                }
                _ => unreachable!(),
            }
        }
        res.reverse();
        res
    }

    fn cyclic_form(&self) -> Vec<Vec<usize>> {
        let mut unchecked = vec![true; self.perm.len()];
        let mut cyclic_form = vec![];

        for i in 0..self.perm.len() {
            if unchecked[i] {
                let mut cycle = vec![i];
                unchecked[i] = false;
                let mut j = i;
                while unchecked[self.perm[j]] {
                    j = self.perm[j];
                    cycle.push(j);
                    unchecked[j] = false;
                }
                if cycle.len() > 1 {
                    cyclic_form.push(cycle);
                }
            }
        }
        cyclic_form.sort();
        cyclic_form
    }
}

impl MulAssign<&mut Permutation> for Permutation {
    fn mul_assign(&mut self, other: &mut Permutation) {
        other.perm.extend(other.perm.len()..self.perm.len());
        self.perm = self.perm.iter().map(|i| other.perm[*i]).collect();
        self.perm.extend(other.perm.iter().take(self.perm.len()));
    }
}

pub fn swap_bit(i: usize, j: usize, b: usize) -> usize {
    let mut b_out = b;
    if i != j {
        let b_i = (b >> i) & 1;
        let b_j = (b >> j) & 1;
        if b_i != b_j {
            b_out &= !((1 << i) | (1 << j));
            b_out |= (b_i << j) | (b_j << i);
        }
    }
    b_out
}

pub fn swap(x: usize, y: usize, n: usize) -> Permutation {
    if x == y {
        Permutation::id(2usize.pow(n as u32))
    } else {
        Permutation::new(
            (0..(2usize.pow(n as u32)))
                .map(|i| swap_bit(n - 1 - x, n - 1 - y, i))
                .collect(),
        )
    }
}

pub fn calc_permutation_matrix(num_qubits: usize, location: Vec<usize>) -> Array2<c64> {
    let max_qubit = location.iter().max().unwrap();
    let num_core_qubits = max_qubit + 1;
    let num_gate_qubits = location.len();

    let mut perm = Permutation::id(2usize.pow(num_core_qubits as u32));
    let mut temp_pos: Vec<usize> = (0..num_gate_qubits).collect();

    for q in 0..num_gate_qubits {
        let mut swap_out = swap(temp_pos[q], location[q], num_core_qubits);
        perm *= &mut swap_out;
        if location[q] < num_gate_qubits {
            temp_pos[location[q]] = temp_pos[q];
        }
    }

    let mut mat = Array2::eye(2usize.pow(num_core_qubits as u32));

    for transposition in perm.transpositions() {
        mat.swap_rows(transposition.0, transposition.1);
    }
    if num_qubits as isize - num_core_qubits as isize > 0 {
        let id = Array2::eye(2usize.pow((num_qubits - num_core_qubits) as u32));
        mat = mat.kron(&id);
    }
    mat
}

/// Permute a unitary so that it spans a circuit of size and is in the correct location
pub fn permute_unitary(
    unitary: ArrayView2<c64>,
    size: usize,
    location: Vec<usize>,
) -> Array2<c64> {
    if size == 0 {
        panic!("Invalid size for permute_unitary");
    }
    if unitary.shape()[0] == 0 || unitary.shape()[1] == 0 {
        panic!("Invalid shape for unitary");
    }
    let normal: Vec<usize> = (0..size).collect();
    if location == normal {
        return unitary.to_owned();
    }
    let id = Array2::eye(2usize.pow((size - location.len()) as u32));
    let permutation_matrix = calc_permutation_matrix(size, location);
    let full_unitary = unitary.kron(&id);
    let permuted = permutation_matrix.matmul(full_unitary.view());
    permuted.matmul(permutation_matrix.t())
}
