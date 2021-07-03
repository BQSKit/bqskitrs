use ndarray::{Array2, Zip};
use num_complex::Complex64;

pub trait SplitComplex {
    fn split_complex(&self) -> (Array2<f64>, Array2<f64>);
}

impl SplitComplex for Array2<Complex64> {
    fn split_complex(&self) -> (Array2<f64>, Array2<f64>) {
        let size = self.shape()[0];
        // Safety: these two arrays are initialized from `self` in the Zip below.
        let mut re = Array2::uninit((size, size));
        let mut im = Array2::uninit((size, size));
        Zip::from(self)
            .and(&mut re)
            .and(&mut im)
            .for_each(|slf, re, im| {
                let re_ptr: *mut f64 = re.as_mut_ptr();
                unsafe { re_ptr.write(slf.re) };
                let im_ptr: *mut f64 = im.as_mut_ptr();
                unsafe { im_ptr.write(slf.im) };
            });
        unsafe { (re.assume_init(), im.assume_init()) }
    }
}
