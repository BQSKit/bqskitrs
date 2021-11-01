use ndarray::{Array2, ArrayView2, CowArray, Ix2, ShapeBuilder};
use num_complex::Complex64;

use cblas_sys::cblas_zgemm;
use cblas_sys::{CblasNoTrans, CblasRowMajor, CblasTrans};

pub trait Matmul {
    fn matmul(&self, other: &ArrayView2<Complex64>) -> Array2<Complex64>;
}

impl Matmul for CowArray<'_, Complex64, Ix2> {
    fn matmul(&self, other: &ArrayView2<Complex64>) -> Array2<Complex64> {
        let ((mut m, a), (_, mut n)) = (self.dim(), other.dim());
        let mut lhs_ = self.view();
        let mut rhs_ = other.view();

        let lhs_s0 = lhs_.strides()[0];
        let rhs_s0 = rhs_.strides()[0];

        let column_major = lhs_s0 == 1 && rhs_s0 == 1;
        let mut v = Vec::with_capacity(m * n);
        let mut out;
        unsafe {
            v.set_len(m * n);
            out = Array2::from_shape_vec_unchecked((m, n).set_f(column_major), v);
        }
        let mut c_ = out.view_mut();
        let both_f = lhs_s0 == 1 && rhs_s0 == 1;
        let mut lhs_trans = CblasNoTrans;
        let mut rhs_trans = CblasNoTrans;
        if both_f {
            // A^t B^t = C^t => B A = C
            let lhs_t = lhs_.reversed_axes();
            lhs_ = rhs_.reversed_axes();
            rhs_ = lhs_t;
            c_ = c_.reversed_axes();
            std::mem::swap(&mut m, &mut n);
        } else if lhs_s0 == 1 && m == a {
            lhs_ = lhs_.reversed_axes();
            lhs_trans = CblasTrans;
        } else if rhs_s0 == 1 && a == n {
            rhs_ = rhs_.reversed_axes();
            rhs_trans = CblasTrans;
        }

        let (m, k) = match lhs_trans {
            CblasNoTrans => lhs_.dim(),
            _ => {
                let (rows, cols) = lhs_.dim();
                (cols, rows)
            }
        };
        let n = match rhs_trans {
            CblasNoTrans => rhs_.raw_dim()[1],
            _ => rhs_.raw_dim()[0],
        };

        // adjust strides, these may [1, 1] for column matrices
        let lhs_stride = std::cmp::max(lhs_.strides()[0] as i32, k as i32);
        let rhs_stride = std::cmp::max(rhs_.strides()[0] as i32, n as i32);
        let c_stride = std::cmp::max(c_.strides()[0] as i32, n as i32);
        let one = r!(1.);
        let zero = r!(0.);
        unsafe {
            cblas_zgemm(
                CblasRowMajor,
                lhs_trans,
                rhs_trans,
                m as i32,
                n as i32,
                k as i32,
                &one as *const Complex64 as *const _,
                lhs_.as_ptr() as *const _,
                lhs_stride as i32,
                rhs_.as_ptr() as *const _,
                rhs_stride as i32,
                &zero as *const Complex64 as *const _,
                c_.as_mut_ptr() as *mut _,
                c_stride as i32,
            )
        };
        out
    }
}

impl Matmul for Array2<Complex64> {
    fn matmul(&self, other: &ArrayView2<Complex64>) -> Array2<Complex64> {
        let ((mut m, a), (_, mut n)) = (self.dim(), other.dim());
        let mut lhs_ = self.view();
        let mut rhs_ = other.view();

        let lhs_s0 = lhs_.strides()[0];
        let rhs_s0 = rhs_.strides()[0];

        let column_major = lhs_s0 == 1 && rhs_s0 == 1;
        let mut v = Vec::with_capacity(m * n);
        let mut out;
        unsafe {
            v.set_len(m * n);
            out = Array2::from_shape_vec_unchecked((m, n).set_f(column_major), v);
        }
        let mut c_ = out.view_mut();
        let both_f = lhs_s0 == 1 && rhs_s0 == 1;
        let mut lhs_trans = CblasNoTrans;
        let mut rhs_trans = CblasNoTrans;
        if both_f {
            // A^t B^t = C^t => B A = C
            let lhs_t = lhs_.reversed_axes();
            lhs_ = rhs_.reversed_axes();
            rhs_ = lhs_t;
            c_ = c_.reversed_axes();
            std::mem::swap(&mut m, &mut n);
        } else if lhs_s0 == 1 && m == a {
            lhs_ = lhs_.reversed_axes();
            lhs_trans = CblasTrans;
        } else if rhs_s0 == 1 && a == n {
            rhs_ = rhs_.reversed_axes();
            rhs_trans = CblasTrans;
        }

        let (m, k) = match lhs_trans {
            CblasNoTrans => lhs_.dim(),
            _ => {
                let (rows, cols) = lhs_.dim();
                (cols, rows)
            }
        };
        let n = match rhs_trans {
            CblasNoTrans => rhs_.raw_dim()[1],
            _ => rhs_.raw_dim()[0],
        };

        // adjust strides, these may [1, 1] for column matrices
        let lhs_stride = std::cmp::max(lhs_.strides()[0] as i32, k as i32);
        let rhs_stride = std::cmp::max(rhs_.strides()[0] as i32, n as i32);
        let c_stride = std::cmp::max(c_.strides()[0] as i32, n as i32);
        let one = r!(1.);
        let zero = r!(0.);
        unsafe {
            cblas_zgemm(
                CblasRowMajor,
                lhs_trans,
                rhs_trans,
                m as i32,
                n as i32,
                k as i32,
                &one as *const Complex64 as *const _,
                lhs_.as_ptr() as *const _,
                lhs_stride as i32,
                rhs_.as_ptr() as *const _,
                rhs_stride as i32,
                &zero as *const Complex64 as *const _,
                c_.as_mut_ptr() as *mut _,
                c_stride as i32,
            )
        };
        out
    }
}

impl Matmul for ArrayView2<'_, Complex64> {
    fn matmul(&self, other: &ArrayView2<Complex64>) -> Array2<Complex64> {
        let ((mut m, a), (_, mut n)) = (self.dim(), other.dim());
        let mut lhs_ = self.view();
        let mut rhs_ = other.view();

        let lhs_s0 = lhs_.strides()[0];
        let rhs_s0 = rhs_.strides()[0];

        let column_major = lhs_s0 == 1 && rhs_s0 == 1;
        let mut v = Vec::with_capacity(m * n);
        let mut out;
        unsafe {
            v.set_len(m * n);
            out = Array2::from_shape_vec_unchecked((m, n).set_f(column_major), v);
        }
        let mut c_ = out.view_mut();
        let both_f = lhs_s0 == 1 && rhs_s0 == 1;
        let mut lhs_trans = CblasNoTrans;
        let mut rhs_trans = CblasNoTrans;
        if both_f {
            // A^t B^t = C^t => B A = C
            let lhs_t = lhs_.reversed_axes();
            lhs_ = rhs_.reversed_axes();
            rhs_ = lhs_t;
            c_ = c_.reversed_axes();
            std::mem::swap(&mut m, &mut n);
        } else if lhs_s0 == 1 && m == a {
            lhs_ = lhs_.reversed_axes();
            lhs_trans = CblasTrans;
        } else if rhs_s0 == 1 && a == n {
            rhs_ = rhs_.reversed_axes();
            rhs_trans = CblasTrans;
        }

        let (m, k) = match lhs_trans {
            CblasNoTrans => lhs_.dim(),
            _ => {
                let (rows, cols) = lhs_.dim();
                (cols, rows)
            }
        };
        let n = match rhs_trans {
            CblasNoTrans => rhs_.raw_dim()[1],
            _ => rhs_.raw_dim()[0],
        };

        // adjust strides, these may [1, 1] for column matrices
        let lhs_stride = std::cmp::max(lhs_.strides()[0] as i32, k as i32);
        let rhs_stride = std::cmp::max(rhs_.strides()[0] as i32, n as i32);
        let c_stride = std::cmp::max(c_.strides()[0] as i32, n as i32);
        let one = r!(1.);
        let zero = r!(0.);
        unsafe {
            cblas_zgemm(
                CblasRowMajor,
                lhs_trans,
                rhs_trans,
                m as i32,
                n as i32,
                k as i32,
                &one as *const Complex64 as *const _,
                lhs_.as_ptr() as *const _,
                lhs_stride as i32,
                rhs_.as_ptr() as *const _,
                rhs_stride as i32,
                &zero as *const Complex64 as *const _,
                c_.as_mut_ptr() as *mut _,
                c_stride as i32,
            )
        };
        out
    }
}
