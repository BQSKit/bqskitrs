use crate::minimizers::{
    CostFunction, DifferentiableResidualFn, HilbertSchmidtResidualFn, ResidualFn,
};
use numpy::PyArray1;
use pyo3::{prelude::*, types::PyTuple};

struct PyResidualFn {
    cost_fn: PyObject,
}

impl PyResidualFn {
    pub fn new(cost_fn: PyObject) -> Self {
        PyResidualFn { cost_fn }
    }
}

impl ResidualFn for PyResidualFn {
    fn get_residuals(&self, params: &[f64]) -> Vec<f64> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let parameters = PyArray1::from_slice(py, params);
        let args = PyTuple::new(py, &[parameters]);
        match self.cost_fn.call_method1(py, "get_cost", args) {
            Ok(val) => val
                .extract::<Vec<f64>>(py)
                .expect("Return type of get_cost was not a float."),
            Err(..) => panic!("Failed to call 'get_cost' on passed CostFunction."), // TODO: make a Python exception?
        }
    }
}

impl DifferentiableResidualFn for PyResidualFn {
    fn get_grad(&self, params: &[f64]) -> Vec<f64> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let parameters = PyArray1::from_slice(py, params);
        let args = PyTuple::new(py, &[parameters]);
        match self.cost_fn.call_method1(py, "get_grad", args) {
            Ok(val) => val
                .extract::<Vec<f64>>(py)
                .expect("Return type of get_grad was not a list of floats."),
            Err(..) => panic!("Failed to call 'get_grad' on passed CostFunction."), // TODO: make a Python exception?
        }
    }
}

#[pyclass(name = "HilbertSchmidtCostFunction", module = "bqskitrs")]
struct PyHilberSchmidtResidualFn {
    cost_fn: HilbertSchmidtResidualFn,
}

fn is_cost_fn_obj<'a>(obj: &'a PyAny) -> PyResult<bool> {
    if obj.hasattr("get_cost")? {
        let get_cost = obj.getattr("get_cost")?;
        if get_cost.is_callable() {
            if obj.hasattr("get_grad")? {
                if obj.getattr("get_grad")?.is_callable() {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

impl<'source> FromPyObject<'source> for CostFunction {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        match ob.extract::<Py<PyHilberSchmidtResidualFn>>() {
            Ok(fun) => Ok(CostFunction::HilbertSchmidt(fun.try_borrow(py)?.cost_fn)),
            Err(..) => {
                if is_cost_fn_obj(ob)? {
                    let fun = PyResidualFn::new(ob.into());
                    Ok(CostFunction::Dynamic(Box::new(fun)))
                } else {
                    panic!("Failed to extract ResidualFn from obj."); // TODO: throw a Python error here.
                }
            }
        }
    }
}
