use std::os::raw::{c_double, c_int, c_void};
use std::sync::Once;

use ceres_sys::ceres::{
    ceres_create_problem, ceres_free_problem, ceres_init, ceres_problem_add_residual_block,
};
use ceres_sys::ceres_solve_silent;

static CERES_INIT: Once = Once::new();

#[repr(C)]
struct ClosureData<'a> {
    cost_fn: &'a mut dyn FnMut(&[f64], &mut [f64], Option<&mut [f64]>),
    nparams: usize,
    nresiduals: usize,
}

extern "C" fn trampoline(
    data: *mut c_void,
    parameters: *mut *mut c_double,
    residuals: *mut c_double,
    jacobian: *mut *mut c_double,
) -> c_int {
    let panic_guard = std::panic::catch_unwind(|| unsafe {
        let data: *mut ClosureData = data.cast();
        let closure_data: &mut ClosureData = data.as_mut().expect("Got NULL `data`");
        let slice = |ptr: *const c_double, len: usize| {
            if ptr.is_null() {
                panic!("Got NULL slice pointer");
            }
            std::slice::from_raw_parts(ptr, len)
        };
        let slice_mut = |ptr: *mut c_double, len: usize| {
            if ptr.is_null() {
                panic!("Got NULL slice pointer");
            }
            std::slice::from_raw_parts_mut(ptr, len)
        };
        if parameters.is_null() {
            panic!("Got NULL parameters");
        }
        let params = std::slice::from_raw_parts(parameters, 1);
        let closure_params = slice(params[0], closure_data.nparams);
        let mut closure_residuals = slice_mut(residuals, closure_data.nresiduals);
        let closure_jac = if jacobian.is_null() {
            None
        } else {
            if jacobian.is_null() {
                panic!("Got NULL jacobian");
            }
            let jacobians = std::slice::from_raw_parts_mut(jacobian, 1);
            if jacobians[0].is_null() {
                None
            } else {
                let clj = slice_mut(jacobians[0], closure_data.nparams * closure_data.nresiduals);
                Some(clj)
            }
        };
        (*closure_data.cost_fn)(&closure_params, &mut closure_residuals, closure_jac);
    });
    match panic_guard {
        // If we don't panic we can return normally
        Ok(_) => 1,
        // Otherwise abort to not cause UB
        Err(_) => std::process::abort(),
    }
}

pub struct CeresSolver {
    num_threads: usize,
    ftol: f64,
    gtol: f64,
    report: bool,
}

impl CeresSolver {
    pub fn new(num_threads: usize, ftol: f64, gtol: f64, report: bool) -> Self {
        CERES_INIT.call_once(|| {
            // Safety: only called once to do onceguard
            unsafe { ceres_init() };
        });
        Self {
            num_threads,
            ftol,
            gtol,
            report,
        }
    }

    pub fn solve<R>(
        &self,
        residual_function: &mut R,
        x0: &mut [f64],
        num_residuals: usize,
        max_iters: usize,
    ) where
        R: FnMut(&[f64], &mut [f64], Option<&mut [f64]>),
    {
        // Safety: ceres_init() already called, FFI wrapper
        let problem = unsafe { ceres_create_problem() };
        let data = &mut ClosureData {
            cost_fn: residual_function,
            nparams: x0.len(),
            nresiduals: num_residuals,
        } as *mut ClosureData as *mut c_void;
        let mut x_ptr = x0.as_mut_ptr();
        // Safety: problem already initialized in new(), data lives as long as function lifetime, as does x_ptr
        unsafe {
            ceres_problem_add_residual_block(
                problem,
                Some(trampoline),
                data,
                None,
                std::ptr::null_mut(),
                num_residuals as i32,
                1,
                &mut x0.len() as *mut usize as *mut i32,
                &mut x_ptr as *mut *mut f64,
            );
        }
        unsafe {
            // Safety: problem initialized in new
            ceres_solve_silent(
                problem,
                max_iters,
                self.num_threads,
                self.ftol,
                self.gtol,
                self.report,
            );
        }
        unsafe {
            // Safety: problem initialized in earlier in this function, originates from ceres_create_problem
            ceres_free_problem(problem);
        }
    }
}
