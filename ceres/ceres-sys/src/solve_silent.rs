use cxx::{type_id, ExternType};

unsafe impl ExternType for crate::ceres::ceres_problem_s {
    type Id = type_id!("ceres_problem_s");
    type Kind = cxx::kind::Opaque;
}

pub use ffi::ceres_solve_silent;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        pub unsafe fn ceres_solve_silent(
            c_problem: *mut ceres_problem_s,
            max_iters: usize,
            num_threads: usize,
            ftol: f64,
            gtol: f64,
            report: bool
        );
        type ceres_problem_s = crate::ceres::ceres_problem_s;
        include!("ceres/c_api.h");
        include!("ceres/ceres.h");
        include!("ceres_wrapper.h");
    }
}
