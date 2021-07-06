use crate::gates::Gradient;
use crate::gates::Optimize;
use crate::gates::Size;
use crate::gates::Unitary;
use crate::{i, r};

use ndarray::Array2;
use ndarray::Array3;
use num_complex::Complex64;

/// IBM's U3 single qubit gate
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct U8Gate();

impl U8Gate {
    pub fn new() -> Self {
        U8Gate {}
    }
}

impl Unitary for U8Gate {
    fn num_params(&self) -> usize {
        8
    }

    fn get_utry(&self, params: &[f64], _constant_gates: &[Array2<Complex64>]) -> Array2<Complex64> {
        let s1 = (params[0]).sin();
        let c1 = (params[0]).cos();
        let s2 = (params[1]).sin();
        let c2 = (params[1]).cos();
        let s3 = (params[2]).sin();
        let c3 = (params[2]).cos();

        let p1 = (i!(1.0) * params[3]).exp();
        let m1 = (i!(-1.0) * params[3]).exp();
        let p2 = (i!(1.0) * params[4]).exp();
        let m2 = (i!(-1.0) * params[4]).exp();
        let p3 = (i!(1.0) * params[5]).exp();
        let m3 = (i!(-1.0) * params[5]).exp();
        let p4 = (i!(1.0) * params[6]).exp();
        let m4 = (i!(-1.0) * params[6]).exp();
        let p5 = (i!(1.0) * params[7]).exp();
        let m5 = (i!(-1.0) * params[7]).exp();

        Array2::from_shape_vec(
            (3, 3),
            vec![
                c1 * c2 * p1,
                s1 * p3,
                c1 * s2 * p4,
                s2 * s3 * m4 * m5 - s1 * c2 * c3 * p1 * p2 * m3,
                c1 * c3 * p2,
                -c2 * s3 * m1 * m5 - s1 * s2 * c3 * p2 * m3 * p4,
                -s1 * c2 * s3 * p1 * m3 * p5 - s2 * c3 * m2 * m4,
                c1 * s3 * p5,
                c2 * c3 * m1 * m2 - s1 * s2 * s3 * m3 * p4 * p5,
            ],
        )
        .unwrap()
    }
}

impl Gradient for U8Gate {
    fn get_grad(&self, params: &[f64], _const_gates: &[Array2<Complex64>]) -> Array3<Complex64> {
        let s1 = (params[0]).sin();
        let c1 = (params[0]).cos();
        let s2 = (params[1]).sin();
        let c2 = (params[1]).cos();
        let s3 = (params[2]).sin();
        let c3 = (params[2]).cos();

        let p1 = (i!(1.0) * params[3]).exp();
        let m1 = (i!(-1.0) * params[3]).exp();
        let p2 = (i!(1.0) * params[4]).exp();
        let m2 = (i!(-1.0) * params[4]).exp();
        let p3 = (i!(1.0) * params[5]).exp();
        let m3 = (i!(-1.0) * params[5]).exp();
        let p4 = (i!(1.0) * params[6]).exp();
        let m4 = (i!(-1.0) * params[6]).exp();
        let p5 = (i!(1.0) * params[7]).exp();
        let m5 = (i!(-1.0) * params[7]).exp();
        Array3::from_shape_vec(
            (3, 3, 8),
            vec![
                // param 0
                -s1 * c2 * p1,
                c1 * p3,
                -s1 * s2 * p4,
                -c1 * c2 * c3 * p1 * p2 * m3,
                -s1 * c3 * p2,
                -c1 * s2 * c3 * p2 * m3 * p4,
                -c1 * c2 * s3 * p1 * m3 * p5,
                -s1 * s3 * p5,
                -c1 * s2 * s3 * m3 * p4 * p5,
                // param 1
                -c1 * s2 * p1,
                r!(0.0),
                c1 * c2 * p4,
                c2 * s3 * m4 * m5 + s1 * s2 * c3 * p1 * p2 * m3,
                r!(0.0),
                s2 * s3 * m1 * m5 - s1 * c2 * c3 * p2 * m3 * p4,
                s1 * s2 * s3 * p1 * m3 * p5 - c2 * c3 * m2 * m4,
                r!(0.0),
                -s2 * c3 * m1 * m2 - s1 * c2 * s3 * m3 * p4 * p5,
                // param 2
                r!(0.0),
                r!(0.0),
                r!(0.0),
                s2 * c3 * m4 * m5 + s1 * c2 * s3 * p1 * p2 * m3,
                -c1 * s3 * p2,
                -c2 * c3 * m1 * m5 + s1 * s2 * s3 * p2 * m3 * p4,
                -s1 * c2 * c3 * p1 * m3 * p5 + s2 * s3 * m2 * m4,
                c1 * c3 * p5,
                -c2 * s3 * m1 * m2 - s1 * s2 * c3 * m3 * p4 * p5,
                // param 3
                i!(1.0) * c1 * c2 * p1,
                r!(0.0),
                r!(0.0),
                -i!(1.0) * s1 * c2 * c3 * p1 * p2 * m3,
                r!(0.0),
                i!(1.0) * c2 * s3 * m1 * m5,
                -i!(1.0) * s1 * c2 * s3 * p1 * m3 * p5,
                r!(0.0),
                -i!(1.0) * c2 * c3 * m1 * m2,
                // param 4
                r!(0.0),
                r!(0.0),
                r!(0.0),
                -i!(1.0) * s1 * c2 * c3 * p1 * p2 * m3,
                i!(1.0) * c1 * c3 * p2,
                -i!(1.0) * s1 * s2 * c3 * p2 * m3 * p4,
                i!(1.0) * s2 * c3 * m2 * m4,
                r!(0.0),
                -i!(1.0) * c2 * c3 * m1 * m2,
                // param 5
                r!(0.0),
                i!(1.0) * s1 * p3,
                r!(0.0),
                i!(1.0) * s1 * c2 * c3 * p1 * p2 * m3,
                r!(0.0),
                i!(1.0) * s1 * s2 * c3 * p2 * m3 * p4,
                i!(1.0) * s1 * c2 * s3 * p1 * m3 * p5,
                r!(0.0),
                i!(1.0) * s1 * s2 * s3 * m3 * p4 * p5,
                // param 6
                r!(0.0),
                r!(0.0),
                i!(1.0) * c1 * s2 * p4,
                -i!(1.0) * s2 * s3 * m4 * m5,
                r!(0.0),
                -i!(1.0) * s1 * s2 * c3 * p2 * m3 * p4,
                i!(1.0) * s2 * c3 * m2 * m4,
                r!(0.0),
                -i!(1.0) * s1 * s2 * s3 * m3 * p4 * p5,
                // param 7
                r!(0.0),
                r!(0.0),
                r!(0.0),
                -i!(1.0) * s2 * s3 * m4 * m5,
                r!(0.0),
                i!(1.0) * c2 * s3 * m1 * m5,
                -i!(1.0) * s1 * c2 * s3 * p1 * m3 * p5,
                i!(1.0) * c1 * s3 * p5,
                -i!(1.0) * s1 * s2 * s3 * m3 * p4 * p5,
            ],
        )
        .unwrap()
    }

    fn get_utry_and_grad(
        &self,
        params: &[f64],
        _const_gates: &[Array2<Complex64>],
    ) -> (Array2<Complex64>, Array3<Complex64>) {
        let s1 = (params[0]).sin();
        let c1 = (params[0]).cos();
        let s2 = (params[1]).sin();
        let c2 = (params[1]).cos();
        let s3 = (params[2]).sin();
        let c3 = (params[2]).cos();

        let p1 = (i!(1.0) * params[3]).exp();
        let m1 = (i!(-1.0) * params[3]).exp();
        let p2 = (i!(1.0) * params[4]).exp();
        let m2 = (i!(-1.0) * params[4]).exp();
        let p3 = (i!(1.0) * params[5]).exp();
        let m3 = (i!(-1.0) * params[5]).exp();
        let p4 = (i!(1.0) * params[6]).exp();
        let m4 = (i!(-1.0) * params[6]).exp();
        let p5 = (i!(1.0) * params[7]).exp();
        let m5 = (i!(-1.0) * params[7]).exp();

        (
            Array2::from_shape_vec(
                (3, 3),
                vec![
                    c1 * c2 * p1,
                    s1 * p3,
                    c1 * s2 * p4,
                    s2 * s3 * m4 * m5 - s1 * c2 * c3 * p1 * p2 * m3,
                    c1 * c3 * p2,
                    -c2 * s3 * m1 * m5 - s1 * s2 * c3 * p2 * m3 * p4,
                    -s1 * c2 * s3 * p1 * m3 * p5 - s2 * c3 * m2 * m4,
                    c1 * s3 * p5,
                    c2 * c3 * m1 * m2 - s1 * s2 * s3 * m3 * p4 * p5,
                ],
            )
            .unwrap(),
            Array3::from_shape_vec(
                (3, 3, 8),
                vec![
                    // param 0
                    -s1 * c2 * p1,
                    c1 * p3,
                    -s1 * s2 * p4,
                    -c1 * c2 * c3 * p1 * p2 * m3,
                    -s1 * c3 * p2,
                    -c1 * s2 * c3 * p2 * m3 * p4,
                    -c1 * c2 * s3 * p1 * m3 * p5,
                    -s1 * s3 * p5,
                    -c1 * s2 * s3 * m3 * p4 * p5,
                    // param 1
                    -c1 * s2 * p1,
                    r!(0.0),
                    c1 * c2 * p4,
                    c2 * s3 * m4 * m5 + s1 * s2 * c3 * p1 * p2 * m3,
                    r!(0.0),
                    s2 * s3 * m1 * m5 - s1 * c2 * c3 * p2 * m3 * p4,
                    s1 * s2 * s3 * p1 * m3 * p5 - c2 * c3 * m2 * m4,
                    r!(0.0),
                    -s2 * c3 * m1 * m2 - s1 * c2 * s3 * m3 * p4 * p5,
                    // param 2
                    r!(0.0),
                    r!(0.0),
                    r!(0.0),
                    s2 * c3 * m4 * m5 + s1 * c2 * s3 * p1 * p2 * m3,
                    -c1 * s3 * p2,
                    -c2 * c3 * m1 * m5 + s1 * s2 * s3 * p2 * m3 * p4,
                    -s1 * c2 * c3 * p1 * m3 * p5 + s2 * s3 * m2 * m4,
                    c1 * c3 * p5,
                    -c2 * s3 * m1 * m2 - s1 * s2 * c3 * m3 * p4 * p5,
                    // param 3
                    i!(1.0) * c1 * c2 * p1,
                    r!(0.0),
                    r!(0.0),
                    -i!(1.0) * s1 * c2 * c3 * p1 * p2 * m3,
                    r!(0.0),
                    i!(1.0) * c2 * s3 * m1 * m5,
                    -i!(1.0) * s1 * c2 * s3 * p1 * m3 * p5,
                    r!(0.0),
                    -i!(1.0) * c2 * c3 * m1 * m2,
                    // param 4
                    r!(0.0),
                    r!(0.0),
                    r!(0.0),
                    -i!(1.0) * s1 * c2 * c3 * p1 * p2 * m3,
                    i!(1.0) * c1 * c3 * p2,
                    -i!(1.0) * s1 * s2 * c3 * p2 * m3 * p4,
                    i!(1.0) * s2 * c3 * m2 * m4,
                    r!(0.0),
                    -i!(1.0) * c2 * c3 * m1 * m2,
                    // param 5
                    r!(0.0),
                    i!(1.0) * s1 * p3,
                    r!(0.0),
                    i!(1.0) * s1 * c2 * c3 * p1 * p2 * m3,
                    r!(0.0),
                    i!(1.0) * s1 * s2 * c3 * p2 * m3 * p4,
                    i!(1.0) * s1 * c2 * s3 * p1 * m3 * p5,
                    r!(0.0),
                    i!(1.0) * s1 * s2 * s3 * m3 * p4 * p5,
                    // param 6
                    r!(0.0),
                    r!(0.0),
                    i!(1.0) * c1 * s2 * p4,
                    -i!(1.0) * s2 * s3 * m4 * m5,
                    r!(0.0),
                    -i!(1.0) * s1 * s2 * c3 * p2 * m3 * p4,
                    i!(1.0) * s2 * c3 * m2 * m4,
                    r!(0.0),
                    -i!(1.0) * s1 * s2 * s3 * m3 * p4 * p5,
                    // param 7
                    r!(0.0),
                    r!(0.0),
                    r!(0.0),
                    -i!(1.0) * s2 * s3 * m4 * m5,
                    r!(0.0),
                    i!(1.0) * c2 * s3 * m1 * m5,
                    -i!(1.0) * s1 * c2 * s3 * p1 * m3 * p5,
                    i!(1.0) * c1 * s3 * p5,
                    -i!(1.0) * s1 * s2 * s3 * m3 * p4 * p5,
                ],
            )
            .unwrap(),
        )
    }
}

impl Size for U8Gate {
    fn get_size(&self) -> usize {
        1
    }
}

impl Optimize for U8Gate {}
