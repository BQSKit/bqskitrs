use std::{ops::MulAssign, vec};

use squaremat::SquareMatrix;

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
            if nx == 2 {
                res.push((x[0], x[1]));
            } else if nx > 2 {
                let (start, end) = x.split_at_mut(1);
                end.reverse();
                let first = start[0];
                for y in end {
                    res.push((first, *y));
                }
            }
        }
        res
    }

    fn cyclic_form(&self) -> Vec<Vec<usize>> {
        let mut unchecked = vec![true; self.perm.len()];
        let mut cyclic_form = vec![];

        for i in 0..self.perm.len() {
            if unchecked[i] {
                let mut cycle = vec![];
                cycle.push(i);
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

fn swap_bit(i: usize, j: usize, b: usize) -> usize {
    let mut b_out = b;
    if i != j {
        let b_i = (b >> i) & 1;
        let b_j = (b >> j) & 1;
        if b_i != b_j {
            b_out &= !((1 << i) | (1 << j));
            b_out |= (b_i << j) | (b_j << i)
        }
    }
    b_out
}

fn swap(x: usize, y: usize, n: usize) -> Permutation {
    Permutation::new(
        (0..(2usize.pow(n as u32)))
            .map(|i| swap_bit(n - 1 - x, n - 1 - y, i))
            .collect(),
    )
}

pub fn calc_permutation_matrix(num_qubits: usize, location: Vec<usize>) -> SquareMatrix {
    let max_qubit = location.iter().max().unwrap();
    let num_core_qubits = max_qubit + 1;
    let num_gate_qubits = location.len();

    let mut perm = Permutation::id(2usize.pow(num_core_qubits as u32));
    let mut temp_pos: Vec<usize> = (0..num_gate_qubits).collect();

    for q in 0..num_gate_qubits {
        perm *= &mut swap(temp_pos[q], location[q], num_core_qubits);
        if location[q] < num_gate_qubits {
            temp_pos[location[q]] = temp_pos[q];
        }
    }

    let mut mat = SquareMatrix::eye(2usize.pow(num_core_qubits as u32));

    for transposition in perm.transpositions() {
        mat.swap_rows(transposition.0, transposition.1);
    }
    if num_qubits - num_core_qubits > 0 {
        let id = SquareMatrix::eye(2usize.pow((num_qubits - num_core_qubits) as u32));
        mat = mat.kron(&id);
    }
    mat
}
